# MIPMatrixTools

[![Build Status](https://github.com/gmlangeloh/MIPMatrixTools.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gmlangeloh/MIPMatrixTools.jl/actions/workflows/CI.yml?query=branch%3Amain)

This package provides tools for defining integer programs in matricial form with the type `IPInstance` and working with classical IP solvers.
In the future, it will also provide a generalization to lattice programming. This repository is part of the [IPGBs project](https://github.com/gmlangeloh/IPGBs.jl).
Its main feature is transforming JuMP models into a matrix form that can be used to solve IPs with Gröbner basis methods. As the IPGBs interface to JuMP
is improved, it will likely be deprecated.

This repository also contains an instance generator for various problems (knapsacks, set cover, set packing, etc) in `CombinatorialOptimizationInstances.jl` and
the instances that were generated for current IPGBs experiments in the `instances` subdirectory.
