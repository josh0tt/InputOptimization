# Input Optimization
<h3>Embedded YouTube Video</h3>
<iframe width="560" height="315" src="https://www.youtube.com/watch?v=hpQK-EyVLx4&list=PLEkwz33miFSsJfw4MfZdJcVMX79RrEs--&index=6" frameborder="0" allowfullscreen></iframe>

Demonstrations of our method in three different simulation environments: 
1) [WaterLily.jl](https://github.com/weymouth/WaterLily.jl)
2) [Aerobench](https://github.com/stanleybak/AeroBenchVVPython)
3) [X-Plane 11](https://www.x-plane.com/product/desktop/)

<p align="center">
  <img alt="Variance" src="https://github.com/josh0tt/InputOptimization/blob/master/img/main.gif" width="100%">
</p>

We also benchmark our method against the orthogonal multisine method proposed by Morelli:
> Morelli, Eugene A. "Optimal input design for aircraft stability and control flight testing." Journal of Optimization Theory and Applications 191.2 (2021): 415-439.


# Instructions
Navigate to `path/to/InputOptimization`
```julia 
] activate .
using InputOptimization
```

To run the experiments with the Aerobench simulator:
```julia
julia> InputOptimization.run_f16_experiments()
```

To run the experiments with WaterLily.jl: 
```julia
julia> InputOptimization.run_cylinder_experiments()
```

To run X-Plane, first open X-Plane and load a new flight. Then
```julia 
julia --threads 1,8
] activate .
using InputOptimization
julia> InputOptimization.run_xplane()
```
and follow instructions in the terminal. Make sure you have the relevant `.sit` file loaded in `/path/to/X-Plane 11/Output/situations/short_cessna.sit`. You can find the one used in the paper [here](https://github.com/josh0tt/InputOptimization/blob/master/data/short_cessna.sit). 

<!-- # X-Plane
<p align="center">
  <img alt="Variance" src="https://github.com/josh0tt/InputOptimization/blob/master/img/main.jpeg" width="100%">
</p>

# Aerobench -->