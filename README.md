# pyTOP

A small 2D FEM/optimization module that will be used to experiment on topology optimization problems (and more generally PDE-constrained problems).

## The FEM module

The FEM module (`fem2d`) allows the user to create a structured triangular mesh. Then, using `IntegralTerm` and `RHSIntegralTerm`) the user can add weak formulation parts into a finite element problem (`Problem`) and obtain a discretized solution. See the [example notebook](https://github.com/paluneau/pyTOP/blob/main/fe_examples.ipynb).

## The optimization module
The optimization module (`optimutils`) allows to use various projected descent methods to solve PDE-constrained optimization problems where the gradient is obtained through the adjoint equations. The base class (`ProjectedDescentLineSearchMethod`) is highly modulable, so it is easy to add new solvers inheriting from it with only a few method overridden. See the [example notebook](https://github.com/paluneau/pyTOP/blob/main/optimutils_example.ipynb)

## Topology optimization

The FEM module can be used to resolve topology optimization problems. In the [TO example notebook](https://github.com/paluneau/pyTOP/blob/main/top_example.ipynb), a classical MBB beam example can be seen. The optimization is carried out with the NLOpt library.

![A topology obtained for 1024 elements](https://github.com/paluneau/pyTOP/blob/main/top2.png)
