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
Genie.config.cors_headers["Access-Control-Allow-Origin"] = "*"
Genie.config.cors_headers["Access-Control-Allow-Headers"] = "Content-Type"
Genie.config.cors_headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
Genie.config.cors_allowed_origins = ["*"]

const DIR_PATH = "upload/"
mkpath(DIR_PATH)

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
    @private pool_created          = false
    @in file                       = "Select Equation"
    @in save_options               = false
    @in save_pool                       = false
    @in reset                      = false
    @in retrain                         = false
    @in var_ranges                 = []
    @in batch_number               = 1
    @out file                           = Dict(
                                        "Initial_dataset_X" => Matrix{Float32}(undef, 1, 1),
                                        "Initial_dataset_y" => Vector{Float32}(undef, 1),
                                        "options" => Dict(),
                                        "num_features" => 0,
                                        "hall_of_fames" => [],
                                        "proposed_points" => DataFrame(),
                                    )
    @out error_message             = ""
    @out result_message            = ""
    @out members                   = ""
    @out saving_options            = false
    @out block_options             = false
    @out symb_df                   = DataFrame()
    @out symb_table                = DataTable(DataFrame())
    @out constrained_vars          = []
    @out constrained_loss_options  = ["Symmetry", "Divergency Type I", "Divergency Type II"]
    @out option_args               = Dict()
    @out dict_to_interphase        = Dict()
    # ---- Loss Functions Definition ----
    @private vars_list             = []
    @private loss_function         = L2DistLoss()
    @in constrained_loss           = "None"
    @in loss                       = "L2DistLoss"
    @in selectedFirst              = "Make a selection"
    @in selectedSecond             = "Make a selection"
    @in use_constrained_loss       = false
    @in lambda                     = 100
    @in div_value                  = 0.0
    @in split                      = 0.1
    @out define_symmetry_vars      = false
    @out define_divergencyI_vars   = false
    @out define_divergencyII_vars  = false
    @out vars_list1                = []     ##Updated when number of       variables is defined
    @out vars_list2                = []
         
    @private new_index             = [-1]
    @private add_new_point_tb      = 0
    @private update_current_data   = 0
    @private remaining_data        = MyDataset( Matrix{Float32}(undef, 0, 0), Vector{Float32}(undef, 0), 0, 0)
    @in train                           = false
    @in add_new_point              = false
    @in adding_new_point           = false
    @in reject_new_point           = false
    @in rejecting_new_point        = false
    @in bin_op1                    = ["+", "-", "*", "/"]
    @in un_op1                     = ["sin", "cos", "exp", "log","sigmoid","nested_exp"]
    @in disagreement_measurement   = "IBMD"
    @in n_populations              = 10
    @in n_iterations               = 10
    @in index                           = Vector{Int}(undef, 1)
    @in new_y                       = Vector{Float32}(undef, 1)
    @in new_X                       = Matrix{Float32}(undef, 1, 1)
    @out yourfile                  = ""
    @out error_message             = ""
    @out use_multiple              = true
    @out training                  = false
    @out see_equations             = false
    @out see_new_point             = false
    @out remaining_pool_created    = false
    @out bin_ops                   = ["+", "-", "*", "/"]
    @out un_ops                    = ["sin", "cos", "exp", "log","sigmoid","nested_exp"]
    @out d_m_ops                   = ["IBMD", "standard deviation"]
    @out maxscore                  = [0.0]
    @out new_point_df              = DataFrame()
    @out new_points_df             = DataFrame()
    @out new_point_table           = DataTable(DataFrame())
    @out new_points_table          = DataTable(DataFrame())
    @out results_table             = DataTable(DataFrame())
    @out remaining_pool            = MyDataset( Matrix{Float32}(undef, 0, 0), Vector{Float32}(undef, 0), 0, 0)
    @out current_data              = MyDataset( Matrix{Float32}(undef, 0, 0), Vector{Float32}(undef, 0), 0, 0)
    
    #################===========Plot Variables================################

    @out scatter_data = [scatter()]
    @out scatter_layout = PlotlyBase.Layout(title="Disagreement Measures: IBMD vs Standard Deviation", height=500,width=500, margin=attr(l=20, r=20, t=45, b=20))
    #################===========Selecting Files================################
    @private df     = DataFrame()
    @out message         = ""
    @out see_trainx = false
    @out table      = DataTable()
    @out upfiles    = readdir("upload/")

    
    @in yourfile       = "Select a file"
    @in label_column   = ""
    @in use_header     = false
    @in select_label   = false
    @in header         = 1
    @out column_names  = []
    @out filepath = ""
    @out datay         = Vector{Float64}(undef, 0)
    @out dataX         = Matrix{Float64}(undef, 0, 0)
    current_datetime = replace(string(ceil(now(),Dates.Minute(1))),":"=>"_") 
    @out linkhref = "/data.json"
    @out dfile_name = ""
    @in dfile = false
    ############==Type of Test Data=================################
    @in group = "v1"
    @out version_1 = true
    @out version_2 = false
    @out web_options = [Dict("label"=>"Testing",
                    "value"=>"v1"),
                    Dict("label"=>"Exploring",
                    "value"=>"v2")]
    ##############=============Search Space================################
    @in number_of_variables = 0
    @out readonly = false
    @out X_1 = []
    @out X_2 = []
    @out X_3 = []
    @out X_4 = []
    @out X_5 = []
    @out X_6 = []
    @in Low_bound_X1 = 0
    @in Low_bound_X2 = 0
    @in Low_bound_X3 = 0
    @in Low_bound_X4 = 0
    @in Low_bound_X5 = 0
    @in Low_bound_X6 = 0
    @in Low_bound_X7 = 0
    @in High_bound_X1 = 10
    @in High_bound_X2 = 10
    @in High_bound_X3 = 10
    @in High_bound_X4 = 10
    @in High_bound_X5 = 10
    @in High_bound_X6 = 10
    @in High_bound_X7 = 10
    @in Number_of_X1 = 20
    @in Number_of_X2 = 20
    @in Number_of_X3 = 20
    @in Number_of_X4 = 20
    @in Number_of_X5 = 20
    @in Number_of_X6 = 20
    @in Number_of_X7 = 20
    @out pool = Matrix{Float64}(undef, 0, 0)
    @in save_pool = false
    @private pool_created = false
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

    
    @onbutton reset begin
        @show "resetting parameters"
        file                       = "Select Equation"
        println("before",vars_list)
        vars_list                  = []
        println("after",vars_list)
        loss_function              = L2DistLoss()
        constrained_loss           = "None"
        loss                       = "L2DistLoss"
        selectedFirst              = "Make a selection"
        selectedSecond             = "Make a selection"
        use_constrained_loss       = false
        lambda                     = 100
        div_value                  = 0.0
        split                      = 0.1
        define_symmetry_vars       = false
        define_divergencyI_vars    = false
        define_divergencyII_vars   = false
        vars_list1                 = []     ##Updated when number of       variables is defined
        vars_list2                 = []
        yourfile       = "Select a file"
        label_column   = ""
        use_header     = false
        select_label   = false
        header         = 1
        column_names  = []
        filepath = ""
        datay         = Vector{Float64}(undef, 0)
        dataX         = Matrix{Float64}(undef, 0, 0)
        dict_to_interphase        = Dict()
        table      = DataTable()
        see_trainx = false
        symb_df = DataFrame()
        symb_table = DataTable()
        scatter_data = [scatter()]
        bin_op1                    = ["+", "-", "*", "/"]
        un_op1                     = ["sin", "cos", "exp", "log"]
        disagreement_measurement   = "IBMD"
        current_data
        n_populations              = 10
        n_iterations               = 10
        index                           = Vector{Int}(undef, 1)
        new_y                       = Vector{Float32}(undef, 1)
        new_X                       = Matrix{Float32}(undef, 1, 1)
        maxscore                  = [0.0]
        new_point_df              = DataFrame()
        new_points_df             = DataFrame()
        new_point_table           = DataTable(DataFrame())
        new_points_table          = DataTable(DataFrame())
        results_table             = DataTable(DataFrame())
        remaining_pool            = MyDataset( Matrix{Float32}(undef, 0, 0), Vector{Float32}(undef, 0), 0, 0)
        current_data              = MyDataset( Matrix{Float32}(undef, 0, 0), Vector{Float32}(undef, 0), 0, 0)
        block_options             = false

        reset = false
    end
    @onchange group begin
        if group == "v1"
            version_1,version_2 = true,false
        elseif group == "v2"
            version_1,version_2 = false,true
        end
    end
    @onchange yourfile begin
        if !reset
            if group == "v1"
            @show "reading dataset"
            filepath                                    = DIR_PATH * yourfile
            @show("filepath: ", filepath)
            remaining_data                              = load_data(filepath)
            @show("remaining_data: ", remaining_data)
            vars_list = ["var$i" for i in 1:remaining_data.num_features]
            vars_list1 = vars_list
            vars_list2 = vars_list
            new_X, new_y, index, remaining_data = samplenewdata(remaining_data, 10)
            file["Initial_dataset_X"] = new_X
            file["Initial_dataset_y"] = new_y
            file["num_features"] = remaining_data.num_features
            print(file)
            current_data                 = MyDataset(transpose(new_X), 
                                                        new_y, 
                                                        remaining_data.num_features, 
                                                        remaining_data.num_data)       
            dict_to_interphase           = Dict("var$i" => current_data.X[i, :] for i in 1:current_data.num_features)
    
            new_points_df = DataFrame([Symbol("var$i") => Float32[] for i in 1:current_data.num_features]...)
            # Add 'label' and 'Status' columns separately
            new_points_df.label = Float32[]
            new_points_df.Status = String[]
    
            #println(dict_to_interphase)
            dict_to_interphase["label"]  = new_y
            table                        = DataTable(describe(DataFrame(dict_to_interphase),:mean, :std, :min, :q25, :q75, :max, :nnonmissing))
                                                                                                    
            see_trainx                   = true
            @show "Dataset read"
            end
        end
    end

    @onchange update_current_data begin
        if !reset
            dict_to_interphase           = Dict("var$i" => current_data.X[i,:] for i in 1:current_data.num_features)
            dict_to_interphase["label"]  = current_data.y
            table                        = DataTable(DataFrame(dict_to_interphase))
        end
    end

    @onchange selectedFirst, selectedSecond begin
        if !reset
            #make it such that selectedFirst is not in the second list and viceversa
            vars_list1 = filter(x->x != selectedSecond,  vars_list)
            vars_list2 = filter(x->x != selectedFirst ,  vars_list)
        end
    end

    @onchange constrained_loss, use_constrained_loss begin
        if !reset
            if constrained_loss == "None" || !use_constrained_loss
                define_symmetry_vars, define_divergencyI_vars, define_divergencyII_vars = false, false, false  
            elseif constrained_loss == "symmetry"
                define_symmetry_vars, define_divergencyI_vars, define_divergencyII_vars = true, false, false
            elseif constrained_loss == "divergencyI"
                define_symmetry_vars, define_divergencyI_vars, define_divergencyII_vars = false, true, false
            elseif constrained_loss == "divergencyII"
                define_symmetry_vars, define_divergencyI_vars, define_divergencyII_vars = false, false, true
            end
            vars_list1 = vars_list
            vars_list2 = vars_list
        end
    end

    @onbutton save_options begin
        @show "Saving settings"
        saving_options = true
        loss_function  = loss == "L2DistLoss" ? L2DistLoss() : 
                                (loss == "L1DistLoss" ? L1DistLoss() : 
                                error("Invalid loss type"))
        const_loss     = nothing
        if use_constrained_loss
            #determine the column number of the selected variables
            print("here!")
            print("selected_first: ", selectedFirst)
            print("vars_list: ", vars_list)
            selectedFirst_index  = findfirst(x->x == selectedFirst , vars_list)
            print("also here!")
            print("First Index: ", selectedFirst_index)
            print("selectedSecond: ", selectedSecond)
            selectedSecond_index = findfirst(x->x == selectedSecond, vars_list)
            print("and also here!")
            print("Second Index: ", selectedSecond_index)
            if define_symmetry_vars
                constrained_vars = [selectedFirst_index, selectedSecond_index]
                const_loss       = select_constraint("symmetry", 
                                            lambda = lambda, 
                                            vars   = [constrained_vars])
            end
            if define_divergencyI_vars
                constrained_vars = [selectedFirst_index]
                const_loss       = select_constraint("divergencyI", 
                                            lambda = lambda, 
                                            vars   = constrained_vars, 
                                            value  = div_value)
            end
            if define_divergencyII_vars
                constrained_vars = [selectedFirst_index, selectedSecond_index]
                const_loss       = select_constraint("divergencyII", 
                                            lambda = lambda, 
                                            vars   = constrained_vars)
            end
        end
        searchspace = pool_created ? pool : remaining_data

        option_args = Dict("binary_operators"         => bin_op1, 
                                 "unary_operators"         => un_op1, 
                                 "npopulations"            => n_populations, 
                                "stateReturn"              => false, 
                                "niterations"              => n_iterations, 
                                "loss"                     => loss_function,
                                "const_loss"               => const_loss, 
                                "searchspace"              => searchspace, 
                                "filename"                 => yourfile, 
                                "split"                    => split, 
                                "disagreement_measure"     => disagreement_measurement, 
                                "use_physical_constraints" => use_constrained_loss)
        args_for_json = copy(option_args)
        args_for_json["loss"] = option_args["loss"] == L2DistLoss() ? "L2DistLoss" : (option_args["loss"] == L1DistLoss() ? "L1DistLoss" : "Unknown Loss")
        delete!(args_for_json,"searchspace")
        delete!(args_for_json, "use_constrained_loss")
        delete!(args_for_json,"split")
        file["options"] = args_for_json
        saving_options = false
        block_options = true
    end
    #@onchange add_new_point_tb begin
    #    if length(list_new_point_tables) < 5
    #        push!(list_new_point_tables, [add_new_point_tb, new_point_table])
    #    else
    #        list_new_point_tables = list_new_point_tables[2:end]
    #        push!(list_new_point_tables, [add_new_point_tb, new_point_table])
    #    end
#
    #end
    @onbutton train begin
        @show "starting training"
        training        = true
        (members ,
        tuple, 
        _, 
        results  , 
        new_index,
        maxscore,
        IBMD_scores,
        std_scores)       = trainSR(option_args, 
                                        data = current_data,
                                        batch_size = batch_number)
        results_df      = DataFrame(results)
        push!(file["hall_of_fames"],members)
        symb_df = DataFrame(tuple)
        results_table   = DataTable(results_df)
        symb_table      = DataTable(symb_df)
        new_X            = reshape(remaining_data.X[new_index,:], :,length(remaining_data.X[1,:]))
        new_point_df    = DataFrame([Symbol("var$i") => new_X[:, i] for i in 1:size(new_X, 2)],)
        new_point_df.score .= maxscore
        scatter_data    = [scatter(
                                x=IBMD_scores,
                                y=std_scores,
                                mode="markers",
                               )]
        new_point_table = DataTable(new_point_df)
        see_new_point   = true        
        training        = false
        train = false
    end

    @onbutton add_new_point begin
        @show "Adding new point to the training set"
        adding_new_point = true
        try
            new_y             = remaining_data.y[new_index]
            new_index = sort(new_index, rev=true)  # Sorting in descending order
            for new_index in new_index
                deleteat!(remaining_data.y, new_index)
            end

            for new_index in new_index
                remaining_data.X = vcat(remaining_data.X[1:new_index-1,:], remaining_data.X[new_index+1:end,:])
                remaining_data.num_data -= 1
            end

        catch error
            println("Error: ", error)
        finally

            addData!(current_data, transpose(new_X), new_y)

            columns = DataFrame([Symbol("var$i") => new_X[:, i] for i in 1:size(new_X, 2)]...)
            columns.label = new_y
            columns.Status = repeat(["Accepted"], length(new_y))

            if ncol(new_points_df) == 0
                new_points_df = columns
            else
                new_points_df = vcat(new_points_df, columns)
            end
            file["proposed_points"] = new_points_df

            new_points_table = DataTable(new_points_df)
            adding_new_point = false
            update_current_data += 1
            add_new_point = false
            see_equations = true
        end
    end
    @onbutton reject_new_point begin
        rejecting_new_point = true
        try
            new_y             = remaining_data.y[new_index]
            new_index = sort(new_index, rev=true)  # Sorting in descending order
            for new_index in new_index
                deleteat!(remaining_data.y, new_index)
            end

            for new_index in new_index
                remaining_data.X = vcat(remaining_data.X[1:new_index-1,:], remaining_data.X[new_index+1:end,:])
                remaining_data.num_data -= 1
            end
        catch error
            println("Error: ", error)
        finally
            columns = DataFrame([Symbol("var$i") => new_X[:, i] for i in 1:size(new_X, 2)]...)
            columns.label = new_y
            columns.Status = repeat(["Rejected"], length(new_y))

            if ncol(new_points_df) == 0
                new_points_df = columns
            else
                new_points_df = vcat(new_points_df, columns)
            end
            file["proposed_points"] = new_points_df
            new_points_table = DataTable(new_points_df)
            rejecting_new_point = false
            add_new_point = false
        end
    end
    @onchange yourfile, select_label, label_column begin
    if group == "v2"
        @show "reading dataset"
        filepath = DIR_PATH * yourfile
        @show("filepath: ", filepath)
            #if using testing mode 
            #file = yourfile
            #else
                df = CSV.read(filepath, DataFrame, header=header)

                #drop columns full of missing
                df = df[:, map(x->!all(ismissing, df[!, x]), propertynames(df))]
                column_names = names(df)
                y = df[:,end]
                #get everything but the last column
                dataX = Matrix{Float64}(df[:,1:end-1])
                number_of_variables = size(dataX)[2]
                if use_header && !select_label
                    vars_list = column_names[1:end-1]
                    vars_list1 = vars_list
                    vars_list2 = vars_list
                elseif use_header && select_label
                    vars_list = column_names
                    vars_list1 = vars_list
                    vars_list2 = vars_list
                elseif !use_header 
                    vars_list = ["X_$(i)" for i in 1:number_of_variables]
                    vars_list1 = vars_list
                    vars_list2 = vars_list
                end
                table = DataTable(df)
            end
    end
    @onchange save_pool begin
        @show "Saving"
        X_1 = range(Low_bound_X1,High_bound_X1,Number_of_X1)
        X_2 = range(Low_bound_X2,High_bound_X2,Number_of_X2)
        X_3 = range(Low_bound_X3,High_bound_X3,Number_of_X3)
        X_4 = range(Low_bound_X4,High_bound_X4,Number_of_X4)
        X_5 = range(Low_bound_X5,High_bound_X5,Number_of_X5)
        X_6 = range(Low_bound_X6,High_bound_X6,Number_of_X6)
        X_7 = range(Low_bound_X7,High_bound_X7,Number_of_X7)
        variables = [X_1,X_2,X_3,X_4,X_5,X_6,X_7]
        vars_list = ["X_1",:"X_2","X_3","X_4","X_5","X_6","X_7"]
        #Eliminate last element until only number_of_variables remaining
        while length(variables) > number_of_variables
            pop!(variables)
            pop!(vars_list)
        end
        vars_list1 = vars_list
        vars_list2 = vars_list
        var_ranges = []
        for i in 1:number_of_variables
            push!(var_ranges, variables[i])
        end
        pool = createPoolMatrix(var_ranges)
        @show "size of the pool: $(size(pool)[2]) possible values"
        pool_created = true
    end
end #app
@page("/app", "app.jl.html")
route("/app", method = POST) do
    files = Genie.Requests.filespayload()
    for f in files
        write(joinpath(DIR_PATH, f[2].name), f[2].data)
    end
    if length(files) == 0
        @info "No file uploaded"
    end
    return "Upload finished"
  end
include("demo.jl")
end #module