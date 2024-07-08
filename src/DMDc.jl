# Adapted from Anthony Corso's DMDc Julia implementation
# https://github.com/ancorso/DMDC/blob/master/dmdc.jl

# Get the normalized singular values from the matrix
function normalized_singular_values(X)
    _, Σ, _ = svd(X)
    cumsum(Σ[2:end].^2) / sum(Σ[2:end].^2)
end

# Get the number of modes to keep for a specified amount of the energy to be maintained.
# Rounds up to the closest even number
function num_modes(Σ, retained_energy)
    r = findfirst(cumsum(Σ[2:end].^2) / sum(Σ[2:end].^2) .> retained_energy)
    Int(round((r+1)/2, RoundUp)*2)
end

function DMDc(Ω, Xp, retained_energy = 0.99; num_modes_override = nothing)
    n = size(Xp, 1)
    q = size(Ω, 1) - n

    # Compute the singular value decomposition of Omega (the input space)
    U, Σ, V = svd(Ω)
    p = (num_modes_override == nothing) ? num_modes(Σ, retained_energy) : num_modes_override
    println("Input space truncation: ", p)
    
    # Truncate the matrices
    U_til, Σ_til, V_til = U[:,1:p], diagm(0 =>Σ[1:p]), V[:, 1:p]

    # Compute this efficient SVD of the output space Xp
    U, Σ, V = svd(Xp)
    r = (num_modes_override == nothing) ? num_modes(Σ, retained_energy) : num_modes_override
    println("Output space truncation: ", r)
    U_hat = U[:,1:r] # note that U_hat' * U_hat \approx I

    U_1 = U_til[1:n, :]
    U_2 = U_til[n+1:n+q, :]
    Σ_inv = inv(Σ_til)

    A = U_hat' * Xp * V_til * Σ_inv * U_1' * U_hat
    B = U_hat' * Xp * V_til * Σ_inv * U_2'

    D, W = eigen(A)

    new_U_hat = (Xp * V_til * Σ_inv) * (U_1' * U_hat)
    ϕ =  new_U_hat * W
    transform = pinv(new_U_hat)

    return A, B, ϕ, W, transform, U_hat#new_U_hat
end

function project_down(data::Matrix{Float64}, U_hat::Matrix{Float64})
    # x̃ = Û' * x
    data_reduced = U_hat' * data
    return data_reduced
end

function project_up(data_reduced::Matrix{Float64}, U_hat::Matrix{Float64})
    # x = Û * x̃
    data = U_hat * data_reduced
    return data
end