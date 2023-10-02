using JuMP

#Linear assignment problems
function generate_lap(n :: Int)
    obj = rand(1:n, n, n)
    model = Model()
    @variable(model, x[1:n, 1:n], Bin)
    @objective(model, Min, sum(obj[i, j] * x[i,j] for i in 1:n, j in 1:n))
    for j in 1:n
        @constraint(model, sum(x[i,j] for i in 1:n) == 1)
    end
    for i in 1:n
        @constraint(model, sum(x[i,j] for j in 1:n) == 1)
    end
    return model, x
end

function repeated_subsets(subsets :: Vector{Set{Int}})
    return any(subsets[i] == subsets[j] for i in 1:length(subsets), j in 1:length(subsets) if i != j) 
end

function is_feasible_set_cover(subsets :: Vector{Set{Int}}, n :: Int)
    return foldl(union, subsets) == Set(1:n)
end

function generate_subsets(n :: Int, m :: Int, p :: Float64)
    subsets = Vector{Set{Int}}()
    for i in 1:m
        subset = Set{Int}()
        for j in 1:n
            if rand() < p
                push!(subset, j)
            end
        end
        push!(subsets, subset)
    end
    return subsets
end

#Set covering problems
#Given n elements (say, the numbers 1:n) and m subsets of these elements, find 
#a minimum number of subsets such that every element is contained in at least 
#one of the chosen subsets.
#p is the probability any given element is in a subset.
function generate_set_cover(n :: Int, m :: Int, p :: Float64)
    subsets = generate_subsets(n, m, p)
    model = Model()
    @variable(model, x[1:m], Bin)
    @objective(model, Min, sum(x[i] for i in 1:m))
    for i in 1:n
        @constraint(model, sum(x[j] for j in 1:m if i in subsets[j]) >= 1)
    end
    return model, x
end

#Set packing problems
#Given n elements (numbers 1:n) and m subsets of these elements, find a maximum
#number of subsets such that no element is contained in more than one of them.
#p is the probability any given element is in a subset.
function generate_set_packing(n :: Int, m :: Int, p :: Float64)
    subsets = generate_subsets(n, m, p)
    model = Model()
    @variable(model, x[1:m], Bin)
    @objective(model, Max, sum(x[i] for i in 1:m))
    for i in 1:n
        @constraint(model, sum(x[j] for j in 1:m if i in subsets[j]) <= 1)
    end
    return model
end

using MIPMatrixTools.IPInstances
using IPGBs.FourTi2
using IPGBs

lap_model, _ = generate_lap(5)
lap = IPInstance(lap_model, infer_binary=false)
gb = groebner(lap)
@show size(gb) gb
gb2 = groebner_basis(lap)
@show size(gb2)