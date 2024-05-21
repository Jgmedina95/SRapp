module App
# set up Genie development environment
using GenieFramework, DataFrames, CSV
using Main.SRutils

@genietools
Genie.config.cors_headers["Access-Control-Allow-Origin"] = "*"
Genie.config.cors_headers["Access-Control-Allow-Headers"] = "Content-Type"
Genie.config.cors_headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
Genie.config.cors_allowed_origins = ["*"]

const DIR_PATH = "upload/"
mkpath(DIR_PATH)
#####TOO COMPLICATED TO HAVE FUNCTIONS COMING FROM SRutils#########
function process_dominating(searchspace, dominating, options_wo_const, disagreement_measure)
    losses = [member.loss for member in dominating]
    new_point, new_index = committee_evaluation(searchspace, dominating, options_wo_const, disagreement_measure)
    members = [string_tree(member.tree, options_wo_const) for member in dominating]
    for member in members
        println(" Equation: $member")
    end
    return members, losses, new_point, new_index
end
function IBMD(x::AbstractArray{T},y::AbstractArray{T}) where T<:Real
    # Calculate the denominator and handle zero cases
    denom = abs.(max.(x, y))
    safe_denom = ifelse.(denom .== 0, 1, denom)  # Replace 0 with 1 in the denominator

    # Calculate the IBMD, using safe_denom to avoid division by zero
    return log2.((abs.(x .- y) ./ safe_denom) .+ 1)
end

function IBMD_disagreements_index(X::AbstractArray{T},comb::AbstractArray) where T<:Real
    #measure IBMD for every combination 
    cum=zeros(size(X)[1])
    for i in 1:size(comb)[1]
         cum+=IBMD(X[:,comb[i][1]],X[:,comb[i][2]])
    end
    #return index of maximum value in cum
    #print the top 10
    #sorted_indexes = sortperm(cum,rev=true)
    println("first 10 indexes of the sorted cum array")

    return cum, findmax(cum)[2]
end
function committee_evaluation(sample::Union{AbstractArray,MyDataset},
                        dominating,
                        options,
                        disagreement_measure)
    println("I got to the committee_evaluation")
    if isa(sample,MyDataset)
        sample_x = sample.X
    else
        sample_x = sample
    end
    trees = [member.tree for member in dominating]
    Results = [eval_tree_array(node,sample_x,options) for node in trees]    #results => [Evaluations,Bool]# 
    arrays=[Results[i][1] for i in 1:size(Results)[1] ]     #extracting the evaluations from the results#
    y=reduce((x,y)-> hcat(x,y),[vectors for vectors in arrays])  #concantenating in one matrix
    if disagreement_measure == "std"  
        scores = std(y,dims=2)./ abs(mean(y,dims=2)) 
        scores= reshape(scores,(size(scores)[1]))
        maxindex= findmax(scores)[2] 
        return sample_x[:,maxindex],maxindex
    end
    println("Just about to start IBDM")
    if disagreement_measure == "IBMD"
        comb = collect(combinations(1:size(y)[2],2))
        scores,maxindex = IBMD_disagreements_index(y,comb)
        #print the top 10

        return sample_x[:,maxindex],maxindex
    end
    #calculating the scores for each datapoint# #disagreement measuremen#
end    

function mean(x)
    sum(x) / length(x)
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
            ("abs" => abs)])
binary_ops = [map_bin[i] for i in binlist]
unary_ops = [map_un[i] for i in unalist]
return binary_ops, unary_ops
end

function trainSR(options; data = nothing)
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
    println("This is the searchspace: $searchspace")
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
        members, losses, new_point, new_index = process_dominating( searchspace,dominating, options_wo_const, disagreement_measure)
    else 
        hall_of_fame = EquationSearch(data.X,data.y,niterations=niterations,options=options_wo_const)
        dominating = calculate_pareto_frontier(data.X,data.y,hall_of_fame[2],options_wo_const)
        members, losses, new_point, new_index = process_dominating(searchspace,dominating, options_wo_const, disagreement_measure)
    end
    print("I got after processing dominating")
    results = Dict(
        "members" => members,
        "losses" => losses,)
    return join(members, "\n"), new_point, results, new_index
end




function createPoolMatrix(vars)
    cartesian_product = collect(Iterators.product(vars...))
    combinations_matrix = hcat([reduce(vcat, combo) for combo in cartesian_product]...)
    return combinations_matrix
end
###############code for demo file ############################
const DEMO_PATH = "demofiles/"
mkpath(DEMO_PATH)

# add reactive code to make the UI interactive
@app begin
    #################===========Selecting Files================################
    @out uploaded = false
    @out message = ""
    @private df = DataFrame()
    @out table = DataTable()
    @out upfiles = readdir("upload/")
    @onchange uploaded begin
        upfiles = readdir("upload/")
    end
    
    @in yourfile = "Select a file"
    @in use_header = false
    @in header = 1
    @in select_label = false
    @in label_column = ""
    @out column_names = []
    @out dataX = Matrix{Float64}(undef, 0, 0)
    @out datay = Vector{Float64}(undef, 0)

    @onchange df begin 
        table = DataTable(df)
        print(df)
    end
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
    ########=========== Symbolic Regression options ==============###########
    @out bin_ops = ["+", "-", "*", "/"]
    @out unary_ops = ["sin", "cos", "exp", "log", "sqrt", "abs"]
    @in bin_op1 = ["+"]
    @out un_op1 = ["sin"]
    @out use_multiple = true
    @in n_populations = 100
    @in n_iterations = 5
    @out training = false
    @out trained = false
    @out show_results = false
    @out show_not_trained_error_message = false

    @out constrained_vars = []
    @out constrained_loss_options = ["Symmetry", "Divergency Type I", "Divergency Type II"]
    @out option_args = Dict()
    @in save_options = false
    @out members = ""
    @out result_message = ""
    @out error_message = ""
    @in var_ranges = []
                    # ---- Loss Functions Definition ----
    @in use_constrained_loss = false
    @in constrained_loss = "None"
    @in loss = "L2DistLoss"
    @private loss_function = L2DistLoss()
    @out define_symmetry_vars = false
    @out define_divergencyI_vars = false
    @out define_divergencyII_vars = false
    @private vars_list = []
    @in lambda = 100
    @out vars_list1 = []                     ##Updated when number of       variables is defined
    @out vars_list2 = []
    @in selectedFirst = "Make a selection"
    @in selectedSecond = "Make a selection"
    @in div_value = 0.0
    @in is_div_value_a_number = true
    @in split = 0.1
    @in do_something = false


    ##############=============Results================################

    @out results_df = DataFrame()
    @out results_table = DataTable(DataFrame())
    @out new_point = []
    @out new_point_table = DataTable(DataFrame())
    @out see_equations = false
    ##########================DEMO Purposes=================###########

    @out demo_training = false
    @in data = MyDataset(Matrix{Float32}(undef, 0, 0), Vector{Float32}(undef, 0), 0, 0)
    @in demonew_X = Matrix{Float32}(undef, 1, 1)
    @in demonew_y = Vector{Float32}(undef, 1)
    @in index = Vector{Int}(undef, 1)
    @out sample_table = DataTable(DataFrame())
    @out df_x = DataFrame()
    @out table_x = DataTable(DataFrame())
    @out sample_df = DataFrame()
    @out see_trainx = true
    @in save_demo_options = false

    
    @onchange constrained_loss, use_constrained_loss begin
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
    @onchange save_options begin
        @show "Saving"
        training=true
        #bin_operator, un_operator = mapops(bin_op1, un_op1)
        #evolution loss
        loss_function = loss == "L2DistLoss" ? L2DistLoss() : (loss == "L1DistLoss" ? L1DistLoss() : error("Invalid loss type"))
        const_loss = nothing
        if use_constrained_loss
            #determine the column number of the selected variables
            selectedFirst_index = findfirst(x->x==selectedFirst, vars_list)
            selectedSecond_index = findfirst(x->x==selectedSecond, vars_list)

            if define_symmetry_vars
                constrained_vars = [selectedFirst_index, selectedSecond_index]
                const_loss = select_constraint("symmetry", lambda=lambda, vars=[constrained_vars])
            end
            if define_divergencyI_vars
                constrained_vars = selectedFirst_index
                const_loss = select_constraint("divergencyI", lambda=lambda, vars=constrained_vars, value = div_value)
            end
            if define_divergencyII_vars
                constrained_vars = [selectedFirst_index, selectedSecond_index]
                const_loss = select_constraint("divergencyII", lambda=lambda, vars=constrained_vars)
            end
        end
        searchspace = pool_created ? pool : demonew_X
        option_args = Dict("binary_operators" => bin_op1, 
                            "unary_operators" => un_op1, 
                            "npopulations" => n_populations, 
                            "stateReturn" => false, 
                            "niterations" => n_iterations, 
                            "loss" => loss_function,
                            "const_loss"=>const_loss, 
                            "searchspace" => searchspace, 
                            "filename"=> yourfile, 
                            "split" => split, 
                            "header"=>use_header, 
                            "use_header"=>use_header, 
                            "select_label"=>select_label, 
                            "label_column"=>label_column, 
                            "disagreement_measure" => "IBMD", 
                            "use_physical_constraints" => use_constrained_loss)
        members, new_point, results = trainSR(option_args)
        results_df = DataFrame(results)
        results_table = DataTable(results_df)
        df = DataFrame((Symbol("var$i") => val for (i,val) in enumerate(new_point)))
        #change column names to Variables and values
        rename!(df, [Symbol("variables"),Symbol("value")])
        new_point_table = DataTable(df)
        see_equations = true
        @show "Training finished"
        @show members
        result_message = "This are the results $members"
        println(result_message)
        trained = true
        training = false

    end
    @onchange show_results begin
        if trained
            @show members
        else 
            @show "Not trained yet"
            error_message = "Not trained yet"
            show_not_trained_error_message = true
        end
    end
    @onchange yourfile, use_header, select_label, label_column begin
        @show "reading dataset"
        filepath = DIR_PATH * yourfile

        if endswith(filepath, r"csv")
            if use_header 
                df = CSV.read(filepath, DataFrame,delim = " ", header=header)
            else
                df = CSV.read(filepath, DataFrame, header=use_header)
            end

            #drop columns full of missing
            df = df[:, map(x->!all(ismissing, df[!, x]), propertynames(df))]
            column_names = names(df)
            if select_label && label_column != ""
                y = df[!,label_column]
                select!(df, Not([label_column]))
                dataX = Matrix{Float64}(df)
                number_of_variables = size(dataX)[2]
            else
                y = df[:,end]
                #get everything but the last column
                dataX = Matrix{Float64}(df[:,1:end-1])
                number_of_variables = size(dataX)[2]
            end
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
    @onchange selectedFirst, selectedSecond begin
        #make it such that selectedFirst is not in the second list and viceversa
        vars_list1 = filter(x->x!=selectedSecond, vars_list)
        vars_list2 = filter(x->x!=selectedFirst, vars_list)
    end
    #@onbutton do_something begin 
    #    Random_variable = rand(1:10)
    #    if Random_variable > 5
    #        @show "Doing something"
    #        println("Im doing something")
    #    end
    #    @show "Doing something"
    #    do_something = false
    #    println("Im doing something")
    #end
end #app


# register a new route and the page that will be
# loaded on access
##@page("/demo","demo.jl.html")
@page("/", "app.jl.html")
route("/", method = POST) do
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
end #module App