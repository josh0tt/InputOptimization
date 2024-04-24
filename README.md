# Input Optimization
<p align="center">
  <img alt="Variance" src="https://github.com/josh0tt/InputOptimization/blob/master/img/main.jpeg" width="100%">
</p>

# Instructions

Navigate to `path/to/InputOptimization`
```julia 
] activate .
using InputOptimization
```

To run the experiments with the Aerobench simulator:
```julia
julia> InputOptimization.run_experiments()
```

To run X-Plane, first open X-Plane and load a new flight. Then
```julia 
julia --threads 1,8
] activate .
using InputOptimization
julia> InputOptimization.run_xplane()
```
and follow instructions in the terminal. Make sure you have the relevant `.sit` file loaded in `/path/to/X-Plane 11/Output/situations/short_cessna.sit`. You can find the one used in the paper [here](https://github.com/josh0tt/InputOptimization/blob/master/data/short_cessna.sit). 

# X-Plane

# Aerobench