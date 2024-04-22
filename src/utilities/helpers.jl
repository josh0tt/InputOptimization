"""
    scale_bounds(scaler, safe_bounds, start_index, end_index)

Scale the bounds using the given scaler.

# Arguments
- `scaler`: A `UnitRangeTransform` object representing the scaler.
- `safe_bounds`: A matrix of safe bounds.
- `start_index`: The starting index.
- `end_index`: The ending index.

# Returns
A scaled matrix of bounds.

"""
function scale_bounds(scaler::UnitRangeTransform{Float64,Vector{Float64}}, safe_bounds::Matrix{Float64}, start_index::Int, end_index::Int)
    lower_bounds = reshape(Vector{Float64}([safe_bounds[li, 1] for li in start_index:end_index]), 1, :)
    upper_bounds = reshape(Vector{Float64}([safe_bounds[ui, 2] for ui in start_index:end_index]), 1, :)

    # Set NaN values to large number
    lower_bounds[isinf.(lower_bounds)] .= -1e100
    upper_bounds[isinf.(upper_bounds)] .= 1e100

    lower_bounds = StatsBase.transform(scaler, reshape(lower_bounds, :, 1))
    upper_bounds = StatsBase.transform(scaler, reshape(upper_bounds, :, 1))

    # Replace large values with Inf
    upper_bounds[upper_bounds.>1e50] .= Inf
    lower_bounds[lower_bounds.<-1e50] .= -Inf

    return reshape(lower_bounds, :, 1), reshape(upper_bounds, :, 1)
end

"""
    estimate_linear_system(Z::Matrix{Float64}, n::Int64)

Estimates the linear system parameters A and B using the given data matrix Z and the number of inputs n.

# Arguments
- `Z::Matrix{Float64}`: The data matrix containing the input-output data.
- `n::Int64`: The number of inputs in the linear system.

# Returns
- `A::Matrix{Float64}`: The estimated system matrix A.
- `B::Matrix{Float64}`: The estimated input matrix B.
"""
function estimate_linear_system(Z::Matrix{Float64}, n::Int64)
    p = size(Z, 2) - 1
    X_prime = Z[1:n, 2:end]

    # Solve the linear system using pseudo-inverse
    Theta = X_prime * pinv(Z[:, 1:p])

    A = Theta[:, 1:n]
    B = Theta[:, n+1:end]

    return A, B
end
