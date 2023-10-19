using JuMP
using Random

using MIPMatrixTools.GBTools

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
    if isempty(subsets)
        return n == 0
    end
    return foldl(union, subsets) == Set(1:n)
end

is_feasible_set_packing(s :: Vector{Set{Int}}, :: Int) = length(s) > 0

function generate_subsets(n :: Int, m :: Int, p :: Float64, feasibility_check)
    subsets = Vector{Set{Int}}()
    while !feasibility_check(subsets, n) || repeated_subsets(subsets)
        empty!(subsets)
        while length(subsets) < m
            subset = Set{Int}()
            for j in 1:n
                if rand() < p
                    push!(subset, j)
                end
            end
            if isempty(subset)
                continue
            end
            push!(subsets, subset)
        end
    end
    return subsets
end

#Set covering problems
#Given n elements (say, the numbers 1:n) and m subsets of these elements, find 
#a minimum number of subsets such that every element is contained in at least 
#one of the chosen subsets.
#p is the probability any given element is in a subset.
function generate_set_cover(n :: Int, m :: Int, p :: Float64)
    subsets = generate_subsets(n, m, p, is_feasible_set_cover)
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
    subsets = generate_subsets(n, m, p, is_feasible_set_packing)
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
function test_lap(n, reps = 1)
    for rep in 1:reps
        lap_model, _ = generate_lap(n)
        lap = IPInstance(lap_model, infer_binary=false)
        gb, t4ti2, _, _, _ = @timed groebner(lap)
        println("LAP & 4ti2 & ", n, " & 0 & 0 & ", rep, " & ", size(gb, 2), " & ", size(gb, 1), " & ", t4ti2)
        gb2, tipgbs, _, _, _ = @timed groebner_basis(lap)
        println("LAP & IPGBs & ", n, " & 0 & 0 & ", rep, " & ", size(gb, 2), " & ", length(gb2), " & ", tipgbs)
    end
end

function test_set_cover(n, m, p, reps = 1)
    Random.seed!(0)
    for rep in 1:reps
        cov_model, _ = generate_set_cover(n, m, p)
        cov = IPInstance(cov_model, infer_binary=true)
        gb, t4ti2, _, _, _ = @timed groebner(cov)
        println("Cover & 4ti2 & ", n, " & ", m, " & ", p, " & ", rep, " & ", size(gb, 2), " & ", size(gb, 1), " & ", t4ti2)
        gb2, tipgbs, _, _, _ = @timed groebner_basis(cov)
        println("Cover & IPGBs & ", n, " & ", m, " & ", p, " & ", rep, " & ", size(gb, 2), " & ", length(gb2), " & ", tipgbs)
        #println("4ti2 GB:")
        #for i in axes(gb, 1)
        #    println(gb[i, :])
        #end
        #println()
        #println("IPGBs GB:")
        #for g in gb2
        #    println(g)
        #end
        #gb3 = GBTools.tovector(gb)
        #println("4ti2 included in IPGBs? ", GBTools.isincluded(gb3, gb2))
        #println("IPGBs included in 4ti2? ", GBTools.isincluded(gb2, gb3))
    end
end

function test_set_packing(n, m, p, reps = 1)
    for rep in 1:reps
        pack_model = generate_set_packing(n, m, p)
        pack = IPInstance(pack_model, infer_binary=false)
        gb, t4ti2, _, _, _ = @timed groebner(pack)
        println("Packing & 4ti2 & ", n, " & ", m, " & ", p, " & ", rep, " & ", size(gb, 2), " & ", size(gb, 1), " & ", t4ti2)
        gb2, tipgbs, _, _, _ = @timed groebner_basis(pack)
        println("Packing & IPGBs & ", n, " & ", m, " & ", p, " & ", rep, " & ", size(gb, 2), " & ", length(gb2), " & ", tipgbs)
    end
end

function compare_4ti2_ipgbs()
    println("Algorithm & Problem & n & m & p & Rep & Vars & GBSize & Time")
    Random.seed!(0)
    for n in 5:5:20
        test_lap(n, 30)
    end
    for n in 5:5:20
        for m in 3:3:12
            for p in 0.05:0.05:0.5
                test_set_cover(n, m, p, 30)
            end
        end
    end
    for n in 5:5:20
        for m in 3:3:12
            for p in 0.05:0.05:0.5
                test_set_packing(n, m, p, 30)
            end
        end
    end
end