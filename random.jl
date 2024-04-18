# Credit to: ma-laforge 
# https://github.com/JuliaTelecom/PhysicalCommunications.jl/blob/master/src/prbs.jl
# PhysicalCommunications: Pseudo-Random Bit Sequence Generators/Checkers
#-------------------------------------------------------------------------------


#==Constants
===============================================================================#

#Integer representation of polynomial x^p1 + x^p2 + x^p3 + x^p4 + 1
_poly(p1::Int) = one(UInt64)<<(p1-1)
_poly(p1::Int, p2::Int) = _poly(p1) + _poly(p2)
_poly(p1::Int, p2::Int, p3::Int, p4::Int) = _poly(p1) + _poly(p2) + _poly(p3) + _poly(p4)

#==Maximum-length Linear-Feedback Shift Register (LFSR) polynomials/taps (XNOR form)
Ref: Alfke, Efficient Shift Registers, LFSR Counters, and Long Pseudo-Random
     Sequence Generators, Xilinx, XAPP 052, v1.1, 1996.==#
const MAXLFSR_POLYNOMIAL = [
	_poly(64,64) #1: not supported
	_poly(64,64) #2: not supported
	_poly(3,2) #3
	_poly(4,3) #4
	_poly(5,3) #5
	_poly(6,5) #6
	_poly(7,6) #7
	_poly(8,6,5,4) #8
	_poly(9,5) #9
	_poly(10,7) #10
	_poly(11,9) #11
	_poly(12,6,4,1) #12
	_poly(13,4,3,1) #13
	_poly(14,5,3,1) #14
	_poly(15,14) #15
	_poly(16,15,13,4) #16
	_poly(17,14) #17
	_poly(18,11) #18
	_poly(19,6,2,1) #19
	_poly(20,17) #20
	_poly(21,19) #21
	_poly(22,21) #22
	_poly(23,18) #23
	_poly(24,23,22,17) #24
	_poly(25,22) #25
	_poly(26,6,2,1) #26
	_poly(27,5,2,1) #27
	_poly(28,25) #28
	_poly(29,27) #29
	_poly(30,6,4,1) #30
	_poly(31,28) #31
	_poly(32,22,2,1) #32
]


#==Types
===============================================================================#
abstract type SequenceGenerator end #Defines algorithm used by sequence() to create a bit sequence
abstract type PRBSGenerator <: SequenceGenerator end #Specifically a pseudo-random bit sequence

#Define supported algorithms:
struct MaxLFSR{LEN} <: PRBSGenerator; end #Identifies a "Maximum-Length LFSR" algorithm

#Define iterator & state objects:
struct MaxLFSR_Iter{LEN,TRESULT} #LFSR "iterator" object
	seed::UInt64 #Initial state (easier to define here than creating state in parallel)
	mask::UInt64 #Store mask value since it cannot easily be statically evaluated.
	len::Int
end
mutable struct MaxLFSR_State{LEN}
	reg::UInt64 #Current state of LFSR register
	bitsrmg::Int #How many bits left to generate
end


#==Constructors
===============================================================================#
"""
    MaxLFSR(reglen::Int)

Construct an object used to identify the Maximum-length LFSR algorithm of a given shift register length, `reglen`.
"""
MaxLFSR(reglen::Int) = MaxLFSR{reglen}()


#==Helper functions:
===============================================================================#
#Find next bit & update state:
function _nextbit(state::MaxLFSR_State{LEN}, polymask::UInt64) where LEN
	msb = UInt64(1)<<(LEN-1) #Statically compiles if LEN is known

	#Mask out all "non-tap" bits:
	reg = state.reg | polymask
	bit = msb
	for j in 1:LEN
		bit = ~xor(reg, bit)
		reg <<= 1
	end
	bit = UInt64((bit & msb) > 0) #Convert resultant MSB to an integer

	state.reg = (state.reg << 1) | bit #Leaves garbage @ bits above LEN
	state.bitsrmg -= 1
	return bit
end


#Core algorithm for sequence() function (no kwargs):
function _sequence(::MaxLFSR{LEN}, seed::UInt64, len::Int, output::DataType) where LEN
    if !(LEN in 3:32)
        throw(ArgumentError("Invalid LFSR register length, $LEN: 3 <= length <= 32"))
    end
    if LEN >= 64
        throw(OverflowError("Cannot build sequence for MaxLFSR{LEN} with LEN=$LEN >= 64."))
    end
    availbits = (UInt64(1) << LEN) - UInt64(1) # Available LFSR bits
    if (seed & availbits) != seed
        throw(OverflowError("seed=$seed does not fit in LFSR with register length of $LEN."))
    end
    if len < 0
        throw(ArgumentError("Invalid sequence length. len must be non-negative"))
    end

    poly = UInt64(MAXLFSR_POLYNOMIAL[LEN])
    mask = ~poly

    return MaxLFSR_Iter{LEN, output}(seed, mask, len)
end



#==Iterator interface:
===============================================================================#

Base.length(iter::MaxLFSR_Iter) = iter.len
Base.eltype(iter::MaxLFSR_Iter{LEN, TRESULT}) where {LEN, TRESULT} = TRESULT
Iterators.IteratorSize(iter::MaxLFSR_Iter) = Base.HasLength()

function Iterators.iterate(iter::MaxLFSR_Iter{LEN, TRESULT}, state::MaxLFSR_State{LEN}) where {LEN, TRESULT}
	if state.bitsrmg < 1
		return nothing
	end
	bit = _nextbit(state, iter.mask)

	return (TRESULT(bit), state)
end

function Iterators.iterate(iter::MaxLFSR_Iter{LEN}) where LEN
	state = MaxLFSR_State{LEN}(iter.seed, iter.len)
	return iterate(iter, state)
end


#==High-level interface
===============================================================================#
"""
    sequence(t::SequenceGenerator; seed::Integer=11, len::Int=-1, output::DataType=Int)

Create an iterable object that defines a bit sequence of length `len`.

Inputs:
  - t: Instance defining type of algorithm used to generate bit sequence.
  - seed: Initial value of register used to build sequence.
  - len: Number of bits in sequence.
  - output: DataType used for sequence elements (typical values are `Int` or `Bool`).

Example returning the first `1000` bits of a PRBS-`31` pattern constructed with the Maximum-length LFSR algorithm seeded with an initial register value of `11`.:

    pattern = collect(sequence(MaxLFSR(31), seed=11, len=1000, output=Bool)).
"""
sequence(t::MaxLFSR; seed::Integer=11, len::Int=-1, output::DataType=Int) =
	_sequence(t, UInt64(seed), len, output)

"""
    sequence_detecterrors(t::SequenceGenerator, v::Array)

Tests validity of bit sequence using sequence generator algorithm.

NOTE: Seeded from first bits of sequence in v.
"""
function sequence_detecterrors(t::MaxLFSR{LEN}, v::Vector{T}) where {LEN, T<:Number}
    if length(v) <= LEN
        throw(ArgumentError("Pattern vector too short to test validity (must be at least > $LEN)"))
    end

    if T != Bool
        for i in 1:length(v)
            if v[i] < 0 || v[i] > 1
                throw(ArgumentError("Sequence value not ∉ [0,1] @ index $i."))
            end
        end
    end

    seed = UInt64(0)
    for i in 1:LEN
        seed = (seed << 1) | UInt64(v[i])
    end

    _errors = similar(v)
    iter = _sequence(t, seed, length(v)-LEN, T)
    state = MaxLFSR_State{LEN}(iter.seed, iter.len)
    for i in (LEN+1):length(_errors)
        (b, state) = iterate(iter, state)
        _errors[i] = convert(T, b != v[i])
    end

    return _errors
end

function markov_sequence(rng::MersenneTwister, len::Int64, m::Int64, p::Float64=0.75)
    # for each of the m control inputs generate a sequence that alternates between 0 and 1. With probability p=0.75 that we remain at the previous value
    sequence = zeros(m, len)
    sequence[:, 1] = rand(rng, [0,1], m)
    for i in 1:m
        for j in 2:len
            if rand(rng) < p
                sequence[i, j] = sequence[i, j-1]
            else
                sequence[i, j] = sequence[i, j-1] == 0 ? 1 : 0
            end
        end
    end
    return sequence        
end

function run_random(problem::InputOptimizationProblem)
    # Create an instance of MaxLFSR for a register length of 31
    lfsr_instance = MaxLFSR(31)

    Z = problem.Z
    n = problem.n
    m = problem.m
    n_t = problem.n_t
    t_horizon = problem.t_horizon
    A_hat = problem.A_hat
    B_hat = problem.B_hat
    max_As = problem.max_As

    Z_unscaled = StatsBase.reconstruct(problem.scaler, Z)
    U_desired = Z_unscaled[n+1:end, end]

    # PRBS using specified seed and length
    # U = Matrix(hcat([collect(sequence(lfsr_instance, seed=rand(problem.rng, 1:100)+i, len=problem.t_horizon, output=Bool)) for i in 1:problem.m]...)')
    
    # Markov Sequence
    U = markov_sequence(problem.rng, problem.t_horizon, problem.m)
    
    # U is size (m, t_horizon)
    # we need to scale each column of U so that the 1s correspond to max_As + U_desired and 0s correspond to U_desired - max_As
    @show size(U)
    @show size(max_As)
    @show size(U_desired)
    U = U .* 2 .* max_As .+ U_desired .- max_As

    control_traj = U
    control_traj = StatsBase.transform(problem.scaler, vcat(zeros(n, t_horizon), control_traj))
    control_traj = control_traj[n+1:end, :]
    Z_planned = zeros(n+m, t_horizon+n_t)
    Z_planned[:, 1:n_t] = Z
    Z_planned[n+1:end, n_t+1:end] = control_traj

    @show size(Z_planned[n+1:end, n_t+1:end])
    @show size(control_traj)
    @show size(A_hat)
    @show size(B_hat)
    @show size(Z_planned)

    for i in n_t:(n_t+t_horizon-1)
        Z_planned[1:n, i+1] .= A_hat * Z_planned[1:n, i] + B_hat*Z_planned[n+1:end, i]
        # Z_planned[1:n, n_t + i] = Z_planned[1:n, n_t + i - 1] + A_hat * Z_planned[1:n, n_t + i - 1] + B_hat * control_traj[i, :]
    end

    @show StatsBase.reconstruct(problem.scaler, Z_planned)[n+1:end, end-1:end]
    return Z_planned
end

