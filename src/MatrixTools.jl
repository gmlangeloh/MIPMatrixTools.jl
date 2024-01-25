module MatrixTools

export hnf_lattice_basis, fiber_solution, normalize_hnf!, basis_to_uhnf, lift_vector

using AbstractAlgebra

const AlgebraInt = AbstractAlgebra.Integers{Int}()

"""
    hnf_lattice_basis(A :: Matrix{Int})

    Return a row basis for the lattice ker(A) computed using the Upper
    Hermite Normal Form. The entries of this basis tend to be smaller than
    those computed directly from kernel(A).
"""
function hnf_lattice_basis(A :: Matrix{Int})
    m, n = size(A)
    mat_A = matrix(AlgebraInt, transpose(A))
    r = rank(mat_A)
    #Transpose and append identity matrix, so that the lattice basis appears
    #as the last few rows / columns of the uhnf.
    tA = hcat(mat_A, identity_matrix(AlgebraInt, n))
    #tA is a n x (m + n) matrix.
    #hnf_cohen is often slightly faster than hnf
    I = identity_matrix(tA, n)
    #Even though there are apparently no guarantees, running hnf_kb! over 64-bit
    #ints does work. Running hnf_cohen! here instead doesn't, though.
    AbstractAlgebra.hnf_kb!(tA, I)
    #The basis is in the last few rows and columns of H
    basis = tA[(r+1):n, (m+1):(n+m)]
    return basis, r #Row basis of the lattice
end

"""
    fiber_solution(A :: Matrix{Int}, b :: Vector{Int}) :: Vector{Int}

    A solution to Ax = b, that is, an element of the fiber of right-hand side
    `b` in the lattice ker(A).
"""
function fiber_solution(A :: Matrix{Int}, b :: Vector{Int}) :: Vector{Int}
    m, n = size(A)
    mat_A = matrix(AlgebraInt, A)
    mat_b = matrix(AlgebraInt, m, 1, b)
    x = AbstractAlgebra.solve(mat_A, mat_b)
    return Int.(reshape(Array(x), n))
end

"""
    normalize_hnf!(H :: Generic.MatSpaceElem{T})

Change `H` to an upper row HNF matrix satisfying the following property:
all entries above a pivot are non-positive and of smaller magnitude than the pivot

Assumes `H` is already in HNF as defined by AbstractAlgebra.jl, that is, it is in
upper row HNF satisfying:
- all entries above a pivot are non-negative and of smaller magnitude than the pivot
"""
function normalize_hnf!(
    H::Generic.MatSpaceElem{T}
) where {T}
    m, n = size(H)
    for i in 1:m
        #Find the pivot in row i
        j = i
        while j <= n && H[i, j] == 0
            j += 1
        end
        if j > n #We reached a row of zeros, we are done.
            break
        end
        #Update rows above the pivot
        for k in 1:(i-1)
            if H[k, j] > 0 #only change positive entries
                H[k, :] -= H[i, :]
            end
        end
    end
end

function basis_to_uhnf(basis :: Generic.MatSpaceElem{T}) where {T}
    uhnf = hnf(basis)
    normalize_hnf!(uhnf)
    return uhnf
end

"""
    is_normalized_hnf(H :: Generic.MatSpaceElem{T})

Return true iff `H` is in normalized HNF form as defined in `normalize_hnf!`.
"""
function is_normalized_hnf(
    H::Generic.MatSpaceElem{T}
)::Bool where {T}
    m, n = size(H)
    for i in 1:m
        j = i + 1
        while j <= n && H[i, j] == 0
            j += 1
        end
        if j > n
            break
        end
        for k in 1:(i-1)
            if H[k, j] > 0 || (H[k, j] < 0 && abs(H[k, j]) >= H[i, j])
                return false
            end
        end
    end
    return true
end

"""
    initial_solution(
    A :: Matrix{Int},
    b :: Vector{Int}
) :: Vector{Int}

Return a solution to instance.A * x == instance.b, dropping the non-negativity
constraints.
"""
function initial_solution(
    A :: Matrix{Int},
    b :: Vector{Int}
) :: Vector{Int}
    mat_A = matrix(AlgebraInt, A)
    mat_b = matrix(AlgebraInt, length(b), 1, b)
    x = AbstractAlgebra.solve(mat_A, mat_b)
    return reshape(convert.(Int, Array(x)), size(A, 2))
end

function lift_partial_solution(
    solution :: Vector{Int},
    rhs :: Vector{Int},
    constraints :: Matrix{Int}
)
    partial_A = constraints[:, 1:length(solution)]
    partial_b = partial_A * solution
    remaining_b = rhs - partial_b
    remaining_A = constraints[:, (length(solution)+1):end]
    A = matrix(AlgebraInt, remaining_A)
    b = matrix(AlgebraInt, length(remaining_b), 1, remaining_b)
    x = AbstractAlgebra.solve(A, b)
    n = size(A, 2)
    remaining_sol = Int.(reshape(Array(x), n))
    return [solution; remaining_sol]
end

"""
    lift_vector(
    v :: Vector{Int},
    projected_basis :: Generic.MatSpaceElem{Int},
    lattice_basis :: Generic.MatSpaceElem{Int}
) :: Vector{Int}

Lift `v` from an (extended) group relaxation to the full problem given by
`instance`.
"""
function lift_vector(
    v :: Vector{Int},
    projected_basis :: Generic.MatSpaceElem{Int},
    lattice_basis :: Generic.MatSpaceElem{Int}
) :: Vector{Int}
    col_basis = transpose(projected_basis)
    coefs = AbstractAlgebra.solve(col_basis, matrix(AlgebraInt, length(v), 1, v))
    full_col_basis = transpose(lattice_basis)
    res = full_col_basis * coefs
    return reshape(Array(res), length(res))
end

end
