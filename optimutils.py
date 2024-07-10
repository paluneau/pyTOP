import numpy as np
import fem2d as fe
import abc

# CLASSES FOR ADJOINT-BASED PDE-CONSTRAINED OPTIMIZATION

class SolutionOperator:

    def __init__(self, problem : fe.Problem):
        self._problem = problem

    # This method should modify the parameters of self._problem using x
    @abc.abstractmethod
    def setParams(self,x):
        pass

    def getMatrix(self, x):
        self.setParams(x)
        self._problem.solve(OnlyAssembly=True)
        return self._problem.getCurrentMatrix()
    
    def getRHS(self, x):
        self.setParams(x)
        self._problem.solve(OnlyAssembly=True)
        return self._problem.getCurrentRHS()
    
    def compute(self, x) -> np.array:
        self.setParams(x)
        return self._problem.solve()
    
class ProjectedDescentLineSearchMethod:

    def __init__(self):
        self._it = 0
        self._lb = np.nan
        self._ub = np.nan
        self._result = None
        self._history = []
        self._historyDepth = 2

    def setBounds(self,lb,ub):
        self._lb = lb
        self._ub = ub

    # f must have signature f(x,U)
    def setObjective(self,f):
        self._f = f

    # g must have signature g(x,U,Uadj)
    def setGradient(self,g):
        self._g = g

    def setHistoryDepth(self,n):
        assert(n>=1)
        self._historyDepth = n

    def setSolutionOp(self,sol:SolutionOperator):
        self._sol = sol
    
    def setAdjointOp(self,sol:SolutionOperator):
        self._adj = sol
    
    def setConvergenceCriteria(self,gtol,steptol,nmax):
        self._gtol = gtol
        self._steptol = steptol
        self._nmax = nmax

    def projectPoint(self,x):
        if not np.isnan(self._ub) :
            x[x>self._ub] = self._ub
        if not np.isnan(self._lb) :
            x[x<self._lb] = self._lb
        return x

    def projectDirection(self,x,d):
        if not np.isnan(self._ub) :
            idxset = np.abs(x-self._ub)<1e-8
            d[idxset] = np.min(np.vstack((d[idxset],np.zeros_like(d[idxset]))),axis=0)
        if not np.isnan(self._lb) :
            idxset = np.abs(x-self._lb)<1e-8
            d[idxset] = np.max(np.vstack((d[idxset],np.zeros_like(d[idxset]))),axis=0)
        return d
    
    def setLineSearch(self,s0,c,t,lsmax):
        self._withLS = True
        self._s0 = s0
        self._c = c
        self._t = t
        self._lsmax = lsmax

    def setWithLineSearch(self,ls):
        self._withLS = ls
    
    # "Private methods" to compute the state/adjoint state
    # By default just calling the solution operators of the class
    # Could be overwritten to compute, for instance, a low-rank approximation
    def solveState(self,x):
        return self._sol.compute(x)
    
    def solveAdjoint(self,x):
        return self._adj.compute(x)

    def updateHistory(self,x,f,g):
        self._history.append((x,f,g))
        if len(self._history)>self._historyDepth:
            self._history.pop(0)
    
    # Specify how to compute the descent direction
    @abc.abstractmethod
    def descentDirection(self):
        pass

    # The main optimization loop. Shouldnt need to change...
    def optimize(self, x0):
        ngrad = np.Infinity
        ngradp = np.Infinity
        step = np.Infinity
        s = self._s0

        x0 = self.projectPoint(x0)

        while ngrad > self._gtol and ngradp > self._gtol and step > self._steptol and self._it < self._nmax:

            # Precompute known quantities at current iterate
            U = self.solveState(x0)
            Uadj = self.solveAdjoint((x0,U))
            fx0 = self._f(x0,U)
            gx0 = self._g(x0,U,Uadj)
            ngrad = np.sqrt(np.sum(gx0**2))

            self.updateHistory(x0,fx0,gx0)

            # Compute projected descent direction
            p = self.descentDirection()
            p = self.projectDirection(x0,p)
            ngradp = np.linalg.norm(p)
            #p = p/np

            # Compute next iterate
            xt = self.projectPoint(x0 + self._s0*p)
            Ut = self.solveState(xt)
            fxt = self._f(xt,Ut)

            # Armijo linesearch
            if self._withLS is not None :
                m = -np.dot(p,gx0)
                s = self._s0
                j = 0
                while (fx0-fxt < s*self._c*m) and j<self._lsmax:
                    xt = self.projectPoint(x0 + s*p)
                    s *= self._t
                    j = j+1
                    Ut = self.solveState(xt)
                    fxt = self._f(xt,Ut)

            step = np.sqrt(np.sum((xt-x0)**2))
            self._it+=1

            # Monitor
            print("-------------------------------------------------------------------------------")
            print(f"n = {self._it}")
            print(f"xn = {xt}")
            print(f"fn = {fxt}")
            #print(f"gn = {gx0.T}")
            #print(f"pn = {p}")
            print(f"sn = {s}")
            print(f"|gn| = {ngrad}")
            print(f"|pn| = {ngradp}")
            print(f"|dxn| = {step}")

            # Update current iterate
            x0 = xt

        self._result = x0
        return x0

class GradientDescent(ProjectedDescentLineSearchMethod):

    def __init__(self):
        super().__init__()

    def descentDirection(self):
        return -1*self._history[-1][2]

class GradientDescentAdjointFree(GradientDescent):

    def __init__(self):
        super().__init__()
        self._associatedAdjoint = None

    def solveState(self,x):
        U = self._sol.compute(x)
        self._associatedAdjoint = -1*U.copy()
        return U

    def solveAdjoint(self,x):
        return self._associatedAdjoint



def PODBasis(snaps, tol):
    snaps_mean = np.mean(snaps,axis=1)
    snaps_cov = (snaps-snaps_mean).T@(snaps-snaps_mean)
    _, SU, VUT = np.linalg.svd(snaps_cov, full_matrices=False)
    i = 1
    ratio = np.Infinity
    while ratio > tol:
        ratio = np.sum(np.diag(SU)[0:i])/np.sum(np.diag(SU))
        i+=1
    return VUT[0:i,:].T


# TODO : réécrire les deux solveurs comme des classes parce que ça manque de modularité là-dedans
def DescentMethodPOD(x0, gtol, steptol, nmax, nsnap, f, grad, probetat, probadjoint, direction=None, lb=np.nan, ub=np.nan, s0=1, armijo=None):
    ngrad = np.Infinity
    ngradp = np.Infinity
    step = np.Infinity
    s = s0
    i = 0
    d = np.shape(x0)[0]
    x_hist = np.zeros((d,nsnap))
    U_hist = np.zeros((probetat._problem._nddls,nsnap))
    Ua_hist = np.zeros((probadjoint._problem._nddls,nsnap))

    # Projection operators
    def project_point(x):
        if not np.isnan(ub) :
            x[x>ub] = ub
        if not np.isnan(lb) :
            x[x<lb] = lb
        return x

    def project_direction(x,d):
        if not np.isnan(ub) :
            idxset = np.abs(x-ub)<1e-8
            d[idxset] = np.min(np.vstack((d[idxset],np.zeros_like(d[idxset]))),axis=0)
        if not np.isnan(lb) :
            idxset = np.abs(x-lb)<1e-8
            d[idxset] = np.max(np.vstack((d[idxset],np.zeros_like(d[idxset]))),axis=0)
        return d

    x0 = project_point(x0)
    gprec = None
    pprec = None
    recomputeBasis = False

    while ngrad > gtol and ngradp > gtol and step > steptol and i < nmax:

        # Precompute known quantities at current iterate
        fx0 = 0
        gx0 = 0
        ngrad = 0
        Phi = 0
        Umean = 0

        if i == nsnap : 
            recomputeBasis = True

        if i < nsnap:
            U = probetat.compute(x0)
            Uadj = probadjoint.compute((x0,U))
            fx0 = f(x0,U)
            gx0 = grad(x0,U,Uadj)
            ngrad = np.sqrt(np.sum(gx0**2))
            x_hist[:,i] = x0
            U_hist[:,i] = U
            Ua_hist[:,i] = Uadj

        if recomputeBasis:
            Phi = PODBasis(U_hist,1e-6)
            Umean = np.mean(U_hist,axis=1)
            recomputeBasis=False

        if i >= nsnap:
            # compute ROM
            Arb = Phi.T@probetat.getmatrix(x0)@Phi
            Frb = Phi.T@(probetat._problem.getCurrentRHS()-probetat.getmatrix(x0)@Umean)
            alph = np.linalg.solve(Arb, Frb)
            U = Phi*alph + Umean

            res = np.linalg.norm(probetat.getmatrix(x0)*U - probetat._problem.getCurrentRHS())
            if res>1e-8:
                recomputeBasis=True

            Uadj = probadjoint.compute((x0,U))
            fx0 = f(x0,U)
            gx0 = grad(x0,U,Uadj)
            ngrad = np.sqrt(np.sum(gx0**2))


        # Compute projected descent direction
        p = direction(x0,gx0) if direction is not None else -1*gx0
        p = project_direction(x0,p)
        ngradp = np.linalg.norm(p)
        #p = p/np

        # Compute next iterate
        xt = project_point(x0 + s0*p)
        # TODO : ici utiliser le modèle réduit si possible
        Ut = probetat.compute(xt)
        fxt = f(xt,Ut)

        # Armijo linesearch
        if armijo is not None :
            m = -np.dot(p,gx0)
            c = armijo[0]
            t = armijo[1]
            s = s0
            j = 0
            while (fx0-fxt < s*c*m) and j<armijo[2]:
                xt = project_point(x0 + s*p)
                s *= t
                j = j+1
                # TODO : ici aussi
                Ut = probetat.compute(xt)
                fxt = f(xt,Ut)

        step = np.sqrt(np.sum((xt-x0)**2))

        i+=1

        # Monitor
        print("-------------------------------------------------------------------------------")
        print(f"n = {i}")
        print(f"xn = {xt}")
        print(f"fn = {fxt}")
        #print(f"gn = {gx0.T}")
        #print(f"pn = {p}")
        print(f"sn = {s}")
        print(f"|gn| = {ngrad}")
        print(f"|pn| = {ngradp}")
        print(f"|dxn| = {step}")

        # Update current iterate
        x0 = xt

    return x0