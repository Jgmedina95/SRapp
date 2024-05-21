module SRutils
include("PhConstraints.jl")
include("SR_with_constraints.jl")
import LossFunctions: MarginLoss,
                    DistanceLoss,
                    SupervisedLoss,
                    ZeroOneLoss,
                    LogitMarginLoss,
                    PerceptronLoss,
                    HingeLoss,
                    L1HingeLoss,
                    L2HingeLoss,
                    SmoothedL1HingeLoss,
                    ModifiedHuberLoss,
                    L2MarginLoss,
                    ExpLoss,
                    SigmoidLoss,
                    DWDMarginLoss,
                    LPDistLoss,
                    L1DistLoss,
                    L2DistLoss,
                    PeriodicLoss,
                    HuberLoss,
                    EpsilonInsLoss,
                    L1EpsilonInsLoss,
                    L2EpsilonInsLoss,
                    LogitDistLoss,
                    QuantileLoss,
                    LogCoshLoss
#include("QBC.jl")
#include("CommitteeEval.jl")

#using .QBC
#using .ConstraintsModule
#using .QBC
#using .Committee
#using .SR_with_constraints

using SymbolicRegression
using SymbolicUtils
import SymbolicRegression: Options, 
                        EquationSearch, 
                        string_tree, 
                        calculate_pareto_frontier,
                        eval_tree_array,
                        node_to_symbolic

import .SRwithConstraints: MyDataset, 
                        load_data, 
                        addData!, 
                        samplenewdata, 
                        process_dominating, 
                        committee_evaluation, 
                        append_one_data_point, 
                        regression_with_constraints, 
                        regression_with_qbc

import .ConstraintsModule: select_constraint
# add your data analysis code
using Combinatorics
import Combinatorics: combinations


export load_data,
        regression_with_constraints,
        regression_with_qbc,
        samplenewdata,
        MyDataset,
        addData!, 
        Options, 
        EquationSearch, 
        string_tree, 
        calculate_pareto_frontier, 
        string_tree, 
        eval_tree_array,
        L2DistLoss,
        L1DistLoss,
        L2MarginLoss,
        L1HingeLoss,
        L2HingeLoss,
        SmoothedL1HingeLoss,
        ModifiedHuberLoss,
        ExpLoss,
        mapOps,
        SymbolicRegression,
        combinations,
        node_to_symbolic,
        select_constraint
        
end #module SRutils