using JuMP
using CPLEX

"""
Simple Round Robin tournament instance.

n : number of teams
match_cost[i, j, k] : cost of match between teams i and j in round k

The objective is to assign each possible match between two teams to a round 
so that the total cost is minimized.
"""
struct SRRInstance
    n :: Int
    match_cost :: Array{Float64, 3}
end

function is_compatible(match :: Tuple{Int, Int}, round :: Int, solution :: Array{Int, 3})
    #Check if any team of this match is already playing in this round in solution
    for i in 1:size(solution, 1)
        j = match[1]
        k = match[2]
        if solution[min(i, j), max(i, j), round] == 1 || solution[min(i, k), max(i, k), round] == 1
            return false
        end
    end
    return true
end

#Generate an initial solution by putting matches in any round as long as
#each team plays exactly once in each round
function initial_solution(instance :: SRRInstance)
    # List all possible matches
    ms = matches(instance)
    # Generate empty solution
    solution = zeros(Int, instance.n, instance.n, instance.n - 1)
    # Now assign each match to the first round where it is compatible
    for (i, j) in ms
        for k in 1:num_rounds(instance)
            if is_compatible((i, j), k, solution)
                solution[min(i, j), max(i, j), k] = 1
                break
            end
        end
    end
    return solution
end

function linearize_solution(solution :: Array{Int, 3}, instance :: SRRInstance)
    linear_sol = Int[]
    for r in 1:num_rounds(instance)
        for i in 1:instance.n
            for j in 1:instance.n
                if i < j
                    push!(linear_sol, solution[i, j, r])
                end
            end
        end
    end
    return linear_sol
end

#File format for instances:
#The first line contains the number n, the number of teams
#Each of the following lines contain the data 
# of a match between two teams in a fixed round
#Each match is represented by 4 space-separated values:
#The first two values are the indices of the teams playing the match
#The third value is the round of the match
#The fourth value is the cost of the match
#Remark: teams and rounds are numbered from 0 to n - 1 and n - 2 resp.
function read_instance(path :: String)
    file = open(path)
    n = parse(Int, readline(file))
    match_cost = zeros(Float64, n, n, n - 1)
    for line in eachline(file)
        i, j, k, c = split(line)
        i, j, k, c = parse(Int, i) + 1, parse(Int, j) + 1, parse(Int, k) + 1, parse(Float64, c)
        match_cost[i, j, k] = c
    end
    return SRRInstance(n, match_cost)
end

function matches(instance :: SRRInstance)
    #Generate all possible matches between two of the n teams
    matches = [(i, j) for i in 1:instance.n, j in 1:instance.n if i < j]
    return matches
end

function matches(i :: Int, instance :: SRRInstance)
    #Generate all possible matches between team i and the other n-1 teams
    matches = []
    for j in 1:instance.n
        if j < i
            push!(matches, (j, i))
        elseif j > i
            push!(matches, (i, j))
        end
    end
    return matches
end

num_matches(instance :: SRRInstance) = (instance.n * (instance.n - 1)) / 2
num_rounds(instance :: SRRInstance) = instance.n - 1

function is_valid(instance :: SRRInstance)
    return instance.n % 2 == 0 && 
    size(instance.match_cost) == (instance.n, instance.n, instance.n - 1)
end

function basic_formulation(instance :: SRRInstance)
    model = Model(CPLEX.Optimizer)
    M = matches(instance)
    nrnds = num_rounds(instance)
    @variable(model, x[M, 1:nrnds] >= 0, Int)
    @objective(model, Min, sum(instance.match_cost[i, j, k] * x[(i, j), k] for (i, j) in M, k in 1:nrnds))
    # Every match is played exactly once
    @constraint(model, [(i, j) in M], sum(x[(i, j), k] for k in 1:nrnds) == 1)
    # Every team plays exactly once in each round
    @constraint(model, [i in 1:instance.n, k in 1:nrnds], sum(x[m, k] for m in matches(i, instance)) == 1)
    return model
end

function solve(instance_path :: String)
    instance = read_instance(instance_path)
    if !is_valid(instance)
        println("Invalid instance")
        return
    end
    model = basic_formulation(instance)
    set_silent(model)
    set_time_limit_sec(model, 3600)
    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        return round(Int, objective_value(model)), true
    elseif has_values(model)
        return round(Int, objective_value(model)), false
    end
end

const INSTANCE_DIRECTORY = "../instances/roundrobin"

function solve_all_instances()
    #List instances in ../instancias-problema2
    for instance in readdir(INSTANCE_DIRECTORY)
        println("Solving instance $instance")
        val, opt = solve("$INSTANCE_DIRECTORY/$instance")
        println(val, " ", opt)
        flush(stdout)
    end
end