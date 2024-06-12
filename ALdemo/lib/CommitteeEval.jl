module Committee
include("QBC.jl")
using SymbolicRegression
using Statistics
using StatsBase
using Combinatorics
using Random
using InvertedIndices

using .QBC
export committee_evaluation, AppendnewData, process_dominating


function process_dominating(searchspace, dominating, options_wo_const, disagreement_measure)
    losses = [member.loss for member in dominating]
    new_point, new_index = Committee.committee_evaluation(searchspace, dominating, options_wo_const, disagreement_measure)
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
    try
    if isa(sample,MyDataset)
        sample_x = sample.X
    else
        sample_x = sample
    end
    catch
        println("Failed at the first try block")
    end
    try
    trees = [member.tree for member in dominating]
    catch
        println("Failed at the second try block")
    end
    try
    Results = [SymbolicRegression.eval_tree_array(node,sample_x,options) for node in trees]    #results => [Evaluations,Bool]# 
    catch
        println("Failed at the third try block")
    end
    try
    arrays=[Results[i][1] for i in 1:size(Results)[1] ]     #extracting the evaluations from the results#
    catch
        println("Failed at the fourth try block")
    end
    try
        y=reduce((x,y)-> hcat(x,y),[vectors for vectors in arrays])  #concantenating in one matrix
    catch
        println("Failed at the fifth try block")
    end
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


function AppendnewData(test_X,test_y,sample_x,sample_y,maxindex)
    test_X = hcat(test_X,sample_x[:,maxindex])
    test_y = vcat(test_y,sample_y[maxindex])
    #remove the sample from the sample set
    sample_x= sample_x[:,[1:maxindex-1; maxindex+1:end]]
    sample_y= sample_y[[1:maxindex-1; maxindex+1:end],:]
    return test_X,test_y,sample_x,sample_y
end

end
