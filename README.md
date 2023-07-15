# pyTOP

A small 2D FEM module that will be used to experiment on on topology optimization problems.

## The FEM module

The FEM module (`fem2d`) allows the user to create a structured triangular mesh. Then, using `IntegralTerm` and `RHSIntegralTerm`) the user can add weak formulation parts into a finite element problem (`Problem`) and obtain a discretized solution. See the [example notebook](https://github.com/paluneau/pyTOP/blob/main/fe_examples.ipynb).

## Topology optimization

The FEM module can be used to resolve topology optimization problems. In the [TO example notebook](https://github.com/paluneau/pyTOP/blob/main/top_example.ipynb), a classical MBB beam example can be seen.
