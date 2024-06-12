module ReactiveDemo
using GenieFramework, DataFrames, CSV
using Genie
using PlotlyBase
using StatsBase
using Statistics
using Dates
using JSON
using Main.SRutils

@genietools
#TODO Add a latex printer so i can show the current equations, and the best equations so far.
const DEMO_PATH = "demofiles/"
mkpath(DEMO_PATH)

function create_dataframe(dominating, symb_members)
    # Collect the data
    equation_indices = ["Equation $(i)" for i in 1:length(dominating)]
    losses = [member.loss for member in dominating]
    symbolics = symb_members
    
    # Construct the DataFrame
    tuple_symb = (
        Index = equation_indices,
        Loss = losses,
        Symbolic = symbolics
    )
    
    return tuple_symb
end

function process_dominating(searchspace, dominating, options_wo_const, disagreement_measure; batch_size = 1)
    losses = [member.loss for member in dominating]
    new_points, new_indexes, maxscore, std_scores,IBMD_scores = committee_evaluation(searchspace, dominating, options_wo_const, disagreement_measure, batch_size)
    members = [string_tree(member.tree, options_wo_const) for member in dominating]
    symb_members = [node_to_symbolic(member.tree,options_wo_const) for member in dominating]
    symb_df = create_dataframe(dominating,members)
    for member in members
        println(" Equation: $member")
    end
    return members, symb_df, losses, new_points, new_indexes, maxscore, std_scores, IBMD_scores
end
function IBMD(x::AbstractArray{T},y::AbstractArray{T}) where T<:Real
    # Calculate the denominator and handle zero cases
    denom = abs.(max.(x, y))
    safe_denom = ifelse.(denom .== 0, 1, denom)  # Replace 0 with 1 in the denominator

    # Calculate the IBMD, using safe_denom to avoid division by zero
    return log2.((abs.(x .- y) ./ safe_denom) .+ 1)
end

function IBMD_disagreements_index(X::AbstractArray{T},comb::AbstractArray, batch_size) where T<:Real
    #measure IBMD for every combination 
    cum=zeros(size(X)[1])
    for i in 1:size(comb)[1]
         cum+=IBMD(X[:,comb[i][1]],X[:,comb[i][2]])
    end
    #return index of maximum value in cum
    #print the top 10
    #sorted_indexes = sortperm(cum,rev=true)
    println("first 10 indexes of the sorted cum array")
    sorted_indices = sortperm(cum, rev=true)
    #print("sorted_indices",sorted_indices)
    top_n_indices = sorted_indices[1:batch_size]
    #print("top_n_indices",top_n_indices)
    
    return top_n_indices,cum
end
function committee_evaluation(sample::Union{AbstractArray,MyDataset},
                        dominating,
                        options,
                        disagreement_measure,
                        batch_size)
    println("I got to the committee_evaluation")

    if isa(sample,MyDataset)
        sample_x = sample.X
    else
        sample_x = sample
    end

    #transpose sample_x
    sample_x = transpose(sample_x)
    trees = [member.tree for member in dominating]
    Results = [eval_tree_array(node,sample_x,options) for node in trees]    #results => [Evaluations,Bool]# 

    arrays=[Results[i][1] for i in 1:size(Results)[1] ]     #extracting the evaluations from the results#

    y=reduce((x,y)-> hcat(x,y),[vectors for vectors in arrays])  #concantenating in one matrix
    if disagreement_measure == "std"  
        scores = std(y,dims=2)./ abs.(mean(y,dims=2)) 
        std_scores= reshape(scores,(size(scores)[1]))
        sorted_indexes = sortperm(std_scores, rev=true)
        maxindexes = sorted_indexes[1:batch_size]
        top_n_scores = std_scores[top_n_indexes]


        _max_indexes, IBMD_scores = IBMD_disagreements_index(y,collect(combinations(1:size(y)[2],2)),batch_size)
        
        return sample_x[:,top_n_indexes],maxindexes, top_n_scores, std_scores, IBMD_scores
    end
    println("Just about to start IBDM")
    if disagreement_measure == "IBMD"
        comb = collect(combinations(1:size(y)[2],2))
        maxindexes, IBMD_scores = IBMD_disagreements_index(y,comb, batch_size)
        maxscores = IBMD_scores[maxindexes]
        _std_score = std(y,dims=2)./ abs.(mean(y,dims=2))
        std_scores = reshape(_std_score,(size(_std_score)[1]))

        return sample_x[:,maxindexes],maxindexes, maxscores, std_scores, IBMD_scores
    end
    #calculating the scores for each datapoint# #disagreement measuremen#
end    

function mapOps(binlist::Vector{String}, unalist::Vector{String})
    map_bin= Dict([("+" => +),
                ("-" => -),
                ("*" => *),
                ("/" => /)])
    map_un = Dict([("sin" => sin),
                ("cos" => cos),
                ("exp" => exp),
                ("log" => log),
                ("sqrt" => sqrt),
                ("abs" => abs),
                ("sigmoid" => sigmoid),
                ("nested_exp" => nested_exp)])
    binary_ops = [map_bin[i] for i in binlist]
    unary_ops = [map_un[i] for i in unalist]
    return binary_ops, unary_ops
end

function trainSR(options; data = nothing,  batch_size = 1)
    binary_ops = get!(options,"binary_operators", ["+", "-", "*", "/"])
    unary_ops = get!(options,"unary_operators", ["sin", "cos"])
    n_populations = get!(options,"npopulations", 100)
    niterations = get!(options,"niterations", 5)
    use_physical_constraints = get!(options,"use_physical_constraints", false)
    constrained_loss = get!(options,"const_loss", nothing)
    loss = get!(options,"loss", "L2DistLoss")
    split = get!(options,"split", 0.1)
    filename = get!(options, "filename", "")
    searchspace = get!(options,"searchspace", Matrix{Float64}(undef, 0, 0)) #what is this default?
    header = get!(options,"header", false)
    bin_ops,un_ops = mapOps(binary_ops, unary_ops)
    disagreement_measure = get!(options,"disagreement_measure", "IBMD")
    data = data === nothing ? load_data(DIR_PATH * filename; header = header) : data
    options_wo_const =  Options(
           binary_operators=bin_ops,
           unary_operators = un_ops,
           npopulations=n_populations,
           loss = loss,
           stateReturn = !use_physical_constraints #if we are using physical constraints, we don't want to return the state
    )
    if use_physical_constraints
        options_w_const = Options(
            binary_operators = bin_ops,
            unary_operators = un_ops,
            npopulations = n_populations,
            stateReturn = true,
            custom_loss_function = constrained_loss)
 
        hall_of_fame =  regression_with_constraints(data.X,data.y,niterations,options_w_const,options_wo_const, split, max_loops=1)
        dominating = calculate_pareto_frontier(data.X,data.y,hall_of_fame,options_wo_const)
        members, symb_members, losses, new_points, new_indexes, maxscores, std_scores, IBMD_scores  = process_dominating( 
                                                                                        searchspace,
                                                                                        dominating,
                                                                                        options_wo_const,
                                                                                        disagreement_measure,
                                                                                        batch_size = batch_size
                                                                                        )
    else 
        hall_of_fame = EquationSearch(data.X,data.y,niterations=niterations,options=options_wo_const)
        dominating = calculate_pareto_frontier(data.X,data.y,hall_of_fame[2],options_wo_const)
        members,symb_members, losses, new_points, new_indexes, maxscores, std_scores, IBMD_scores = process_dominating(searchspace,
                                                                                        dominating,
                                                                                        options_wo_const,
                                                                                        disagreement_measure,
                                                                                        batch_size=batch_size)
    end
    print("I got after processing dominating")
    results = Dict(
        "members" => members,
        "losses" => losses,)
    #print(IBMD_scores, std_scores)
    return join(members, "\n"), symb_members, new_points, results, new_indexes, maxscores, IBMD_scores, std_scores
end

function find_point_index(matrix::Matrix{Float32}, point::Vector{Float32})::Union{Int, Nothing}
    if size(matrix, 1) != length(point)
        error("The point must have the same number of rows as the matrix.")
    end

    for i in 1:size(matrix, 2)
        if all(matrix[:, i] .== point)  # Check if the i-th column matches the point
            return i  # Return the index if a match is found
        end
    end
    return nothing  # Return nothing if no match is found
end
function sigmoid(x)
    return 1 / (1 + exp(-x))
end
function nested_exp(x)
    return exp(exp(x))
end

@app begin

    #############DEMO Options#############
    @private demo_pool_created          = false
    @in demo_file                       = "Select Equation"
    @in demo_save_options               = false
    @in save_pool                       = false
    @in demo_reset                      = false
    @in retrain                         = false
    @in demo_var_ranges                 = []
    @in demo_batch_number               = 1
    @out file                           = Dict(
                                        "Initial_dataset_X" => Matrix{Float32}(undef, 1, 1),
                                        "Initial_dataset_y" => Vector{Float32}(undef, 1),
                                        "options" => Dict(),
                                        "num_features" => 0,
                                        "hall_of_fames" => [],
                                        "proposed_points" => DataFrame(),
                                    )
    @out demo_error_message             = ""
    @out demo_result_message            = ""
    @out demo_members                   = ""
    @out demo_saving_options            = false
    @out demo_block_options             = false
    @out demo_symb_df                   = DataFrame()
    @out demo_symb_table                = DataTable(DataFrame())
    @out demo_constrained_vars          = []
    @out demo_constrained_loss_options  = ["Symmetry", "Divergency Type I", "Divergency Type II"]
    @out demo_option_args               = Dict()
    @out demo_dict_to_interphase        = Dict()
    # ---- Loss Functions Definition ----
    @private demo_vars_list             = []
    @private demo_loss_function         = L2DistLoss()
    @in demo_constrained_loss           = "None"
    @in demo_loss                       = "L2DistLoss"
    @in demo_selectedFirst              = "Make a selection"
    @in demo_selectedSecond             = "Make a selection"
    @in demo_use_constrained_loss       = false
    @in demo_lambda                     = 100
    @in demo_div_value                  = 0.0
    @in demo_split                      = 0.1
    @out demo_define_symmetry_vars      = false
    @out demo_define_divergencyI_vars   = false
    @out demo_define_divergencyII_vars  = false
    @out demo_vars_list1                = []     ##Updated when number of       variables is defined
    @out demo_vars_list2                = []
         
    @private demo_new_index             = [-1]
    @private demo_add_new_point_tb      = 0
    @private demo_update_current_data   = 0
    @private demo_remaining_data        = MyDataset( Matrix{Float32}(undef, 0, 0), Vector{Float32}(undef, 0), 0, 0)
    @in demo_train                           = false
    @in demo_add_new_point              = false
    @in demo_adding_new_point           = false
    @in demo_reject_new_point           = false
    @in demo_rejecting_new_point        = false
    @in demo_bin_op1                    = ["+", "-", "*", "/"]
    @in demo_un_op1                     = ["sin", "cos", "exp", "log","sigmoid","nested_exp"]
    @in demo_disagreement_measurement   = "IBMD"
    @in demo_n_populations              = 10
    @in demo_n_iterations               = 10
    @in index                           = Vector{Int}(undef, 1)
    @in demonew_y                       = Vector{Float32}(undef, 1)
    @in demonew_X                       = Matrix{Float32}(undef, 1, 1)
    @out demo_yourfile                  = ""
    @out demo_error_message             = ""
    @out demo_use_multiple              = true
    @out demo_training                  = false
    @out demo_see_equations             = false
    @out demo_see_new_point             = false
    @out demo_remaining_pool_created    = false
    @out demo_bin_ops                   = ["+", "-", "*", "/"]
    @out demo_un_ops                    = ["sin", "cos", "exp", "log","sigmoid","nested_exp"]
    @out demo_d_m_ops                   = ["IBMD", "standard deviation"]
    @out demo_maxscore                  = [0.0]
    @out demo_new_point_df              = DataFrame()
    @out demo_new_points_df             = DataFrame()
    @out demo_new_point_table           = DataTable(DataFrame())
    @out demo_new_points_table          = DataTable(DataFrame())
    @out demo_results_table             = DataTable(DataFrame())
    @out demo_remaining_pool            = MyDataset( Matrix{Float32}(undef, 0, 0), Vector{Float32}(undef, 0), 0, 0)
    @out demo_current_data              = MyDataset( Matrix{Float32}(undef, 0, 0), Vector{Float32}(undef, 0), 0, 0)
    
    #################===========Plot Variables================################

    @out demo_scatter_data = [scatter()]
    @out demo_scatter_layout = PlotlyBase.Layout(title="Disagreement Measures: IBMD vs Standard Deviation", height=500,width=500, margin=attr(l=20, r=20, t=45, b=20))
    #################===========Selecting Files================################
    @private demo_df     = DataFrame()
    @out message         = ""
    @out demo_see_trainx = false
    @out demo_table      = DataTable()
    @out demo_upfiles    = readdir("demofiles/")

    
    @in yourfile       = "Select a file"
    @in label_column   = ""
    @in use_header     = false
    @in select_label   = false
    @in header         = 1
    @out column_names  = []
    @out demo_filepath = ""
    @out datay         = Vector{Float64}(undef, 0)
    @out dataX         = Matrix{Float64}(undef, 0, 0)
    current_datetime = replace(string(ceil(now(),Dates.Minute(1))),":"=>"_") 
    @out linkhref = "/data.json"
    @out dfile_name = ""
    @in dfile = false
    @onbutton dfile begin  
        print("file\n",file)
        ## do some processing and write to a file
        write("./public/state_$(current_datetime).json", JSON.json(file))
        sleep(2)
        ## change the link url to trigger the file download script. The URL will still point to 
        ## the same file, but the link href will have a different anchor after the #
        linkhref = "/state_$(current_datetime).json#($(rand(1)))"
        dfile=false
    end

    @mounted """
    setTimeout(() => {
        var linkToWatch = document.getElementById('dynamicLink');
        if (!linkToWatch) {
            console.error('Link element not found');
            return;
        }

        var observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.type === 'attributes' && mutation.attributeName === 'href') {
                    console.log('Href changed to:', linkToWatch.href);
                    linkToWatch.click();
                }
            });
        });

        observer.observe(linkToWatch, { attributes: true });
    }, 2000); // Delay in milliseconds
    """

    
    @onbutton demo_reset begin
        @show "resetting parameters"
        demo_file                       = "Select Equation"
        println("before",demo_vars_list)
        demo_vars_list                  = []
        println("after",demo_vars_list)
        demo_loss_function              = L2DistLoss()
        demo_constrained_loss           = "None"
        demo_loss                       = "L2DistLoss"
        demo_selectedFirst              = "Make a selection"
        demo_selectedSecond             = "Make a selection"
        demo_use_constrained_loss       = false
        demo_lambda                     = 100
        demo_div_value                  = 0.0
        demo_split                      = 0.1
        demo_define_symmetry_vars       = false
        demo_define_divergencyI_vars    = false
        demo_define_divergencyII_vars   = false
        demo_vars_list1                 = []     ##Updated when number of       variables is defined
        demo_vars_list2                 = []
        yourfile       = "Select a file"
        label_column   = ""
        use_header     = false
        select_label   = false
        header         = 1
        column_names  = []
        demo_filepath = ""
        datay         = Vector{Float64}(undef, 0)
        dataX         = Matrix{Float64}(undef, 0, 0)
        demo_dict_to_interphase        = Dict()
        demo_table      = DataTable()
        demo_see_trainx = false
        demo_symb_df = DataFrame()
        demo_symb_table = DataTable()
        demo_scatter_data = [scatter()]
        demo_bin_op1                    = ["+", "-", "*", "/"]
        demo_un_op1                     = ["sin", "cos", "exp", "log"]
        demo_disagreement_measurement   = "IBMD"
        demo_current_data
        demo_n_populations              = 10
        demo_n_iterations               = 10
        index                           = Vector{Int}(undef, 1)
        demonew_y                       = Vector{Float32}(undef, 1)
        demonew_X                       = Matrix{Float32}(undef, 1, 1)
        demo_maxscore                  = [0.0]
        demo_new_point_df              = DataFrame()
        demo_new_points_df             = DataFrame()
        demo_new_point_table           = DataTable(DataFrame())
        demo_new_points_table          = DataTable(DataFrame())
        demo_results_table             = DataTable(DataFrame())
        demo_remaining_pool            = MyDataset( Matrix{Float32}(undef, 0, 0), Vector{Float32}(undef, 0), 0, 0)
        demo_current_data              = MyDataset( Matrix{Float32}(undef, 0, 0), Vector{Float32}(undef, 0), 0, 0)
        demo_block_options             = false

        demo_reset = false
    end
    @onchange demo_file begin
        if !demo_reset
            @show "reading dataset"
            demo_filepath                                    = DEMO_PATH * demo_file
            demo_remaining_data                              = load_data(demo_filepath)
            demo_vars_list = ["var$i" for i in 1:demo_remaining_data.num_features]
            demo_vars_list1 = demo_vars_list
            demo_vars_list2 = demo_vars_list
            demonew_X, demonew_y, index, demo_remaining_data = samplenewdata(demo_remaining_data, 10)
            file["Initial_dataset_X"] = demonew_X
            file["Initial_dataset_y"] = demonew_y
            file["num_features"] = demo_remaining_data.num_features
            print(file)
            demo_current_data                 = MyDataset(transpose(demonew_X), 
                                                        demonew_y, 
                                                        demo_remaining_data.num_features, 
                                                        demo_remaining_data.num_data)       
            demo_dict_to_interphase           = Dict("var$i" => demo_current_data.X[i, :] for i in 1:demo_current_data.num_features)
    
            demo_new_points_df = DataFrame([Symbol("var$i") => Float32[] for i in 1:demo_current_data.num_features]...)
            # Add 'label' and 'Status' columns separately
            demo_new_points_df.label = Float32[]
            demo_new_points_df.Status = String[]
    
            #println(demo_dict_to_interphase)
            demo_dict_to_interphase["label"]  = demonew_y
            demo_table                        = DataTable(describe(DataFrame(demo_dict_to_interphase),:mean, :std, :min, :q25, :q75, :max, :nnonmissing))
                                                                                                    
            demo_see_trainx                   = true
            @show "Dataset read"
        end
    end

    @onchange demo_update_current_data begin
        if !demo_reset
            demo_dict_to_interphase           = Dict("var$i" => demo_current_data.X[i,:] for i in 1:demo_current_data.num_features)
            demo_dict_to_interphase["label"]  = demo_current_data.y
            demo_table                        = DataTable(DataFrame(demo_dict_to_interphase))
        end
    end

    @onchange demo_selectedFirst, demo_selectedSecond begin
        if !demo_reset
            #make it such that selectedFirst is not in the second list and viceversa
            demo_vars_list1 = filter(x->x != demo_selectedSecond,  demo_vars_list)
            demo_vars_list2 = filter(x->x != demo_selectedFirst ,  demo_vars_list)
        end
    end

    @onchange demo_constrained_loss, demo_use_constrained_loss begin
        if !demo_reset
            if demo_constrained_loss == "None" || !demo_use_constrained_loss
                demo_define_symmetry_vars, demo_define_divergencyI_vars, demo_define_divergencyII_vars = false, false, false  
            elseif demo_constrained_loss == "symmetry"
                demo_define_symmetry_vars, demo_define_divergencyI_vars, demo_define_divergencyII_vars = true, false, false
            elseif demo_constrained_loss == "divergencyI"
                demo_define_symmetry_vars, demo_define_divergencyI_vars, demo_define_divergencyII_vars = false, true, false
            elseif demo_constrained_loss == "divergencyII"
                demo_define_symmetry_vars, demo_define_divergencyI_vars, demo_define_divergencyII_vars = false, false, true
            end
            demo_vars_list1 = demo_vars_list
            demo_vars_list2 = demo_vars_list
        end
    end

    @onbutton demo_save_options begin
        @show "Saving settings"
        demo_saving_options = true
        demo_loss_function  = demo_loss == "L2DistLoss" ? L2DistLoss() : 
                                (demo_loss == "L1DistLoss" ? L1DistLoss() : 
                                error("Invalid loss type"))
        demo_const_loss     = nothing
        if demo_use_constrained_loss
            #determine the column number of the selected variables
            print("here!")
            print("demo_selected_first: ", demo_selectedFirst)
            print("demo_vars_list: ", demo_vars_list)
            demo_selectedFirst_index  = findfirst(x->x == demo_selectedFirst , demo_vars_list)
            print("also here!")
            print("First Index: ", demo_selectedFirst_index)
            print("demo_selectedSecond: ", demo_selectedSecond)
            demo_selectedSecond_index = findfirst(x->x == demo_selectedSecond, demo_vars_list)
            print("and also here!")
            print("Second Index: ", demo_selectedSecond_index)
            if demo_define_symmetry_vars
                demo_constrained_vars = [demo_selectedFirst_index, demo_selectedSecond_index]
                demo_const_loss       = select_constraint("symmetry", 
                                            lambda = demo_lambda, 
                                            vars   = [demo_constrained_vars])
            end
            if demo_define_divergencyI_vars
                demo_constrained_vars = [demo_selectedFirst_index]
                demo_const_loss       = select_constraint("divergencyI", 
                                            lambda = demo_lambda, 
                                            vars   = demo_constrained_vars, 
                                            value  = demo_div_value)
            end
            if demo_define_divergencyII_vars
                demo_constrained_vars = [demo_selectedFirst_index, demo_selectedSecond_index]
                demo_const_loss       = select_constraint("divergencyII", 
                                            lambda = demo_lambda, 
                                            vars   = demo_constrained_vars)
            end
        end
        demo_searchspace = demo_pool_created ? demo_pool : demo_remaining_data

        demo_option_args = Dict("binary_operators"         => demo_bin_op1, 
                                 "unary_operators"         => demo_un_op1, 
                                 "npopulations"            => demo_n_populations, 
                                "stateReturn"              => false, 
                                "niterations"              => demo_n_iterations, 
                                "loss"                     => demo_loss_function,
                                "const_loss"               => demo_const_loss, 
                                "searchspace"              => demo_searchspace, 
                                "filename"                 => demo_yourfile, 
                                "split"                    => demo_split, 
                                "disagreement_measure"     => demo_disagreement_measurement, 
                                "use_physical_constraints" => demo_use_constrained_loss)
        args_for_json = copy(demo_option_args)
        args_for_json["loss"] = demo_option_args["loss"] == L2DistLoss() ? "L2DistLoss" : (demo_option_args["loss"] == L1DistLoss() ? "L1DistLoss" : "Unknown Loss")
        delete!(args_for_json,"searchspace")
        delete!(args_for_json, "use_constrained_loss")
        delete!(args_for_json,"split")
        file["options"] = args_for_json
        demo_saving_options = false
        demo_block_options = true
    end
    #@onchange demo_add_new_point_tb begin
    #    if length(demo_list_new_point_tables) < 5
    #        push!(demo_list_new_point_tables, [demo_add_new_point_tb, demo_new_point_table])
    #    else
    #        demo_list_new_point_tables = demo_list_new_point_tables[2:end]
    #        push!(demo_list_new_point_tables, [demo_add_new_point_tb, demo_new_point_table])
    #    end
#
    #end
    @onbutton demo_train begin
        @show "starting training"
        demo_training        = true
        (demo_members ,
        demo_tuple, 
        _, 
        demo_results  , 
        demo_new_index,
        demo_maxscore,
        IBMD_scores,
        std_scores)       = trainSR(demo_option_args, 
                                        data = demo_current_data,
                                        batch_size = demo_batch_number)
        demo_results_df      = DataFrame(demo_results)
        push!(file["hall_of_fames"],demo_members)
        demo_symb_df = DataFrame(demo_tuple)
        demo_results_table   = DataTable(demo_results_df)
        demo_symb_table      = DataTable(demo_symb_df)
        demonew_X            = reshape(demo_remaining_data.X[demo_new_index,:], :,length(demo_remaining_data.X[1,:]))
        demo_new_point_df    = DataFrame([Symbol("var$i") => demonew_X[:, i] for i in 1:size(demonew_X, 2)],)
        demo_new_point_df.score .= demo_maxscore
        demo_scatter_data    = [scatter(
                                x=IBMD_scores,
                                y=std_scores,
                                mode="markers",
                               )]
        demo_new_point_table = DataTable(demo_new_point_df)
        demo_see_new_point   = true        
        demo_training        = false
        demo_train = false
    end

    @onbutton demo_add_new_point begin
        @show "Adding new point to the training set"
        demo_adding_new_point = true
        try
            demonew_y             = demo_remaining_data.y[demo_new_index]
            demo_new_index = sort(demo_new_index, rev=true)  # Sorting in descending order
            for new_index in demo_new_index
                deleteat!(demo_remaining_data.y, new_index)
            end

            for new_index in demo_new_index
                demo_remaining_data.X = vcat(demo_remaining_data.X[1:new_index-1,:], demo_remaining_data.X[new_index+1:end,:])
                demo_remaining_data.num_data -= 1
            end

        catch error
            println("Error: ", error)
        finally

            addData!(demo_current_data, transpose(demonew_X), demonew_y)

            columns = DataFrame([Symbol("var$i") => demonew_X[:, i] for i in 1:size(demonew_X, 2)]...)
            columns.label = demonew_y
            columns.Status = repeat(["Accepted"], length(demonew_y))

            if ncol(demo_new_points_df) == 0
                demo_new_points_df = columns
            else
                demo_new_points_df = vcat(demo_new_points_df, columns)
            end
            file["proposed_points"] = demo_new_points_df

            demo_new_points_table = DataTable(demo_new_points_df)
            demo_adding_new_point = false
            demo_update_current_data += 1
            demo_add_new_point = false
            demo_see_equations = true
        end
    end
    @onbutton demo_reject_new_point begin
        demo_rejecting_new_point = true
        try
            demonew_y             = demo_remaining_data.y[demo_new_index]
            demo_new_index = sort(demo_new_index, rev=true)  # Sorting in descending order
            for new_index in demo_new_index
                deleteat!(demo_remaining_data.y, new_index)
            end

            for new_index in demo_new_index
                demo_remaining_data.X = vcat(demo_remaining_data.X[1:new_index-1,:], demo_remaining_data.X[new_index+1:end,:])
                demo_remaining_data.num_data -= 1
            end
        catch error
            println("Error: ", error)
        finally
            columns = DataFrame([Symbol("var$i") => demonew_X[:, i] for i in 1:size(demonew_X, 2)]...)
            columns.label = demonew_y
            columns.Status = repeat(["Rejected"], length(demonew_y))

            if ncol(demo_new_points_df) == 0
                demo_new_points_df = columns
            else
                demo_new_points_df = vcat(demo_new_points_df, columns)
            end
            file["proposed_points"] = demo_new_points_df
            demo_new_points_table = DataTable(demo_new_points_df)
            demo_rejecting_new_point = false
            demo_add_new_point = false
        end
    end

end #app
@page("/demo","demo.jl.html")
end #module