import numpy as np
import matplotlib.pyplot as plt
import fem2d as fe
import optimutils
from singleton_decorator import singleton
from time import time

if __name__ == '__main__':

    # Mesh
    nx = 32
    ny = 16
    mesh = fe.Mesh(nx, ny, 0, 0, 1, 2)
    nelem = mesh.getNelems()
    nddl = mesh.getNddlsVec()
    refElem = fe.RefElement(mesh)

    # Boundary conditions
    dirichletnodes = np.sort(list(range(0,mesh.getNnodes(),2*nx+1)))
    dirichletddls = np.sort(mesh.getDDLnumerVec()[dirichletnodes,:].flatten())
    ess = np.zeros_like(dirichletddls)

    #neumannnodes = np.array([65,98,131,164]) # 8x16
    neumannnodes = np.array([64,129,194,259]) #16x32
    neumannddls = np.sort(mesh.getDDLnumerVec()[neumannnodes,:].flatten())
    nat = np.array([0,-0.25,0,-0.25,0,-0.25,0,-0.25])

    # Declare elasticity problem
    elas_solver = fe.Problem(mesh,2,"State equation",keepElemMat=False)
    elas_solver.setDirichletBC(dirichletddls,ess)
    elas_solver.setNeumannBC(neumannddls,nat)
    DivSigmaU = fe.ElasticityTerm(refElem)
    elas_solver.addTerm(DivSigmaU)

    # Declare adjoint solver 
    adjoint_solver = fe.Problem(mesh,2,"Adjoint equation",keepElemMat=False)

    # Add the boundary conditions
    adjoint_solver.setDirichletBC(dirichletddls,ess)
    adjoint_solver.setNeumannBC(neumannddls,-0.5*nat)

    # Add the weak formulation terms
    DivSigmaU2 = fe.ElasticityTerm(refElem)
    adjoint_solver.addTerm(DivSigmaU2)
    dJdu = fe.RHSAdjointCompliance(refElem)
    adjoint_solver.addTermRHS(dJdu)

    # Integral formulation of the gradient
    gradientint = fe.DoubleContraction422(refElem)

    # Determine fixed DOFs
    fixed = np.sort(list(np.arange(0,nelem,2*nx))+list(np.arange(0,nelem,2*nx)+1)+[30,31,62,63,94,95])#pour 16x8
    free = np.ones(nelem)
    free[fixed] = 0
    free = np.arange(nelem)[free.astype("bool")].astype("int")
    d = free.shape[0]

    # Declare constraint + objective
    def Volume(x):
        rho = np.ones(nelem)
        rho[free] = x
        return np.dot((0.5*(1/ny)**2)*np.ones(nelem),rho) - 1.2

    def dVolume():
        return (0.5*(1/ny)**2)*np.ones(d)

    def Compliance(x,U):
        rho = np.ones(nelem)
        rho[free] = x
        DivSigmaU.setParams(100*rho**3,0.3)
        elas_solver.solve(OnlyAssembly=True)
        return 0.5*elas_solver.getCurrentMatrix().dot(U).dot(U)

    def dCompliance(x,U,Uadj):
        grad = np.zeros(d)
        rho = np.ones(nelem)
        rho[free] = x
        gradientint.setParams(3*100*rho**2,0.3,0.5*U+Uadj,U)
        for i in range(d):
            grad[i] = gradientint.integrate(free[i])
        return grad

    peno = 0.01
    objective = lambda x, U: Compliance(x,U) + peno*Volume(x)**2/2
    gradobjective = lambda x, U, Uadj: dCompliance(x,U,Uadj) + peno*Volume(x)*dVolume()

    # Solution operators
    @singleton
    class DensityToDisplacement(optimutils.SolutionOperator):
        def __init__(self):
            super().__init__(elas_solver,4)

        def setParams(self,x):
            rho = np.ones(nelem)
            rho[free] = x
            DivSigmaU.setParams(100*rho**3,0.3)

    @singleton
    class DensityToAdjoint(optimutils.SolutionOperator):
        def __init__(self):
            super().__init__(adjoint_solver,4)

        def setParams(self,x):
            rho = np.ones(nelem)
            rho[free] = x[0]
            DivSigmaU2.setParams(100*rho**3,0.3)
            dJdu.setParams(100*rho**3,0.3,x[1])


    probetat = DensityToDisplacement()
    probadjoint = DensityToAdjoint()

    # Optimization
    #optimizer = optimutils.GradientDescentWithPODROM(nddl,10,1e-9,0.1)
    optimizer = optimutils.GradientDescentPODErABLS(nddl,10,1e-9,0.1,0.90,tolFullStart=5*(0.5)**30)
    #optimizer = optimutils.GradientDescentComplianceAdjointFree()
    #optimizer = optimutils.GradientDescentWithGSROM(nddl,10,0.1)
    #optimizer = optimutils.SteepestDescentPollakRibiereAdjointFree()
    optimizer.setBounds(1e-4,1)
    optimizer.setObjective(objective)
    optimizer.setGradient(gradobjective)
    optimizer.setSolutionOp(probetat)
    optimizer.setAdjointOp(probadjoint)
    optimizer.setConvergenceCriteria(1e-6,1e-15,1000)
    optimizer.setLineSearch(5,0.9,0.5,30,ell1=6,ell2=35) # s0 = 5 nmax = 20
    t0 = time()
    solopt = optimizer.optimize(np.ones(d))
    t1 = time()
    print(f"optimization time = {t1-t0}")
    print(f"number of iterations = {optimizer._it}")
    print(f"number of calls to FOM assembly = {elas_solver._nAssembly}")
    print(f"number of calls to FOM solve = {elas_solver._nResolution}")
    print(f"total time of FOM assembly = {elas_solver._tAssembly}")
    print(f"total time of FOM solve = {elas_solver._tResolution}")
    print(f"number of reconstruction of RB = {optimizer._nSVD}")
    print(f"number of calls to ROM solve = {optimizer._nROMSolve}")
    print(f"total time of constructing RB = {optimizer._tSVD}")
    print(f"total time of ROM solve = {optimizer._tROMSolve}")

    rho_steep = np.ones(nelem)
    rho_steep[free] = solopt
    mesh.displayDiscontinuousByElementField(1-rho_steep)


