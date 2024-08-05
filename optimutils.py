import numpy as np
import fem2d as fe
import abc
from time import time

# CLASSES FOR ADJOINT-BASED PDE-CONSTRAINED OPTIMIZATION

class SolutionOperator:

    def __init__(self, problem : fe.Problem,Nproc):
        self._problem = problem
        self.Nproc = Nproc

    # This method should modify the parameters of self._problem using x
    @abc.abstractmethod
    def setParams(self,x):
        pass

    def getMatrix(self, x):
        self.setParams(x)
        self._problem.solve(OnlyAssembly=True,Nproc=self.Nproc)
        return self._problem.getCurrentMatrix()
    
    def getRHS(self, x):
        self.setParams(x)
        self._problem.solve(OnlyAssembly=True,Nproc=self.Nproc)
        return self._problem.getCurrentRHS()
    
    def compute(self, x) -> np.array:
        self.setParams(x)
        return self._problem.solve(Nproc=self.Nproc)
    
class ProjectedDescentLineSearchMethod:

    def __init__(self):
        self._it = 0
        self._lb = np.nan
        self._ub = np.nan
        self._s0 = 1
        self._withLS = False
        self._result = None
        self._history = []
        self._historyDepth = 2
        self._state = 0 # 0 = at initial point, 1=next iterate , 2 = in the linesearch loop
        self._lsit = 0
        self._increaseStep = False
        self._currentStep = self._s0
        self._backtrack = True

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
    
    def setLineSearch(self,s0,c,t,lsmax,backtrack=True):
        self._withLS = True
        self._s0 = s0
        self._c = c
        self._t = t
        self._lsmax = lsmax
        self._backtrack = backtrack

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

    def acceptStep(self,x0,xt,fx0,fxt,p,gx0,s):
        m = -np.dot(p,gx0)
        return fx0-fxt >= s*self._c*m
    
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
            self._state = 0
            U = self.solveState(x0)
            Uadj = self.solveAdjoint((x0,U))
            fx0 = self._f(x0,U)
            gx0 = self._g(x0,U,Uadj)
            ngrad = np.sqrt(np.sum(gx0**2))
            self.updateHistory(x0,fx0,gx0)

            # Reset step length for backtracking line search
            if self._backtrack:
                self._currentStep = self._s0

            # Compute projected descent direction
            p = self.descentDirection()
            p = self.projectDirection(x0,p)
            ngradp = np.linalg.norm(p)
            if self._withLS and ngradp > 1e-15:
                p = p/ngradp

            # Compute next iterate
            self._state = 1
            xt = self.projectPoint(x0 + self._currentStep*p)
            Ut = self.solveState(xt)
            fxt = self._f(xt,Ut)

            # Armijo linesearch
            if self._withLS :
                self._state = 2
                s = self._currentStep
                if self._increaseStep:
                    s *= (1/self._t)
                    self._increaseStep = False
                self._lsit = 0
                while (not self.acceptStep(x0,xt,fx0,fxt,p,gx0,s)) and self._lsit<self._lsmax:
                    xt = self.projectPoint(x0 + s*p)
                    s *= self._t
                    self._lsit += 1
                    Ut = self.solveState(xt)
                    fxt = self._f(xt,Ut)
            
                self._currentStep = s

                #if s<1e-5: #if we get under the lsmax tolerance
                #    self._currentStep*=10

            step = np.sqrt(np.sum((xt-x0)**2))
            self._it+=1

            # Monitor
            print("-------------------------------------------------------------------------------")
            print(f"n = {self._it}")
            #print(f"xn = {xt}")
            print(f"fn = {fxt}")
            #print(f"gn = {gx0.T}")
            #print(f"pn = {p}")
            print(f"sn = {s}")
            print(f"|gn| = {ngrad}")
            print(f"|pn| = {ngradp}")
            print(f"|dxn| = {step}")
            print(f"#ls = {self._lsit}")

            # Update current iterate
            x0 = xt

        self._result = x0
        return x0
    
class ComplianceAdjointFreeMethod(ProjectedDescentLineSearchMethod):

    def __init__(self):
        super().__init__()
        self._associatedAdjoint = None

    def solveState(self,x):
        U = self._sol.compute(x)
        self._associatedAdjoint = -1*U.copy()
        return U

    def solveAdjoint(self,x):
        return self._associatedAdjoint

class GradientDescent(ProjectedDescentLineSearchMethod):

    def __init__(self):
        super().__init__()

    def descentDirection(self):
        return -1*self._history[-1][2]
    
class GradientDescentComplianceAdjointFree(ComplianceAdjointFreeMethod):

    def __init__(self):
        super().__init__()

    def descentDirection(self):
        return -1*self._history[-1][2]

class SteepestDescentPollakRibiereAdjointFree(ComplianceAdjointFreeMethod):

    def __init__(self):
        super().__init__()
        self.setHistoryDepth(2)

    def descentDirection(self):
        g = self._history[-1][2]
        gprec = 0
        beta = 0
        if self._it >= 2:
            gprec = self._history[-2][2]
            beta = np.dot(g,g-gprec)/np.dot(gprec,gprec)
        return -g + beta*gprec
    
class GradientDescentWithPODROM(GradientDescent):
    def __init__(self,size,RBsize,RBtol,resTol):
        super().__init__()
        self._RBSize = RBsize
        self._RBtol = RBtol
        self._needNewRB = True
        self._firstTime = True
        self._V = None
        self._snaps = np.zeros((size,self._RBSize))
        self._resTol = resTol
        self._nSVD = 0
        self._nROMSolve = 0
        self._tSVD = 0
        self._tROMSolve = 0


    def setRBSize(self,size):
        self._RBSize = size

    def PODBasis(self):
        nddl = self._snaps.shape[0]
        snaps_mean = np.reshape(np.mean(self._snaps,axis=1),(nddl,1))
        snaps_cov = (self._snaps-snaps_mean)
        t0 = time()
        VU, SU, _ = np.linalg.svd(snaps_cov, full_matrices=False)
        t1 = time()
        self._tSVD += t1-t0
        self._nSVD += 1
        i = 1
        ratio = 0
        while ratio < 1-self._RBtol:
            ratio = np.sum(SU[0:i])/np.sum(SU)
            i+=1
        print(f"Reduced basis size : {i} (energy ratio = {ratio})")
        return VU[:,0:i]
    
    def solveState(self,x):
        ddl = self._snaps.shape[0]
        U = None
        if self._it < self._RBSize:
            U = self._sol.compute(x)
            if self._state == 0:
                self._snaps[:,self._it] = U
        else:
            if self._needNewRB:
                if not self._firstTime:
                    U = self._sol.compute(x)
                    self._snaps[:,0:-1] = self._snaps[:,1:]
                    self._snaps[:,-1] = U
                self._V = self.PODBasis()
                self._needNewRB = False
                self._firstTime = False
            K = self._sol.getMatrix(x)
            F = self._sol.getRHS(x)

            t0 = time()
            Krb = self._V.T@K@self._V
            snaps_mean = np.reshape(np.mean(self._snaps,axis=1),(ddl,1))
            Frb = self._V.T@F - (self._V.T@K@snaps_mean).flatten()
            Urb = np.linalg.solve(Krb,Frb)
            Uapp = self._V@Urb + snaps_mean.flatten()
            self._nROMSolve += 1
            t1 = time()
            self._tROMSolve += t1-t0
            freeDDLs = self._sol._problem._freeDDLs
            res = np.linalg.norm(K[freeDDLs][:,freeDDLs]@Uapp[freeDDLs]-F[freeDDLs])/np.linalg.norm(F[freeDDLs])
            if res > self._resTol:
                self._needNewRB = True
                if U is None:
                    U = Uapp.copy()
                print(f"Recompute reduced basis (res={res})")
            else:
                print(f"Reduced basis valid (res={res})")
                U = Uapp.copy()

        self._associatedAdjoint = -1*U.copy()
        return U

    def solveAdjoint(self,x):
        return self._associatedAdjoint
    

class GradientDescentWithGSROM(GradientDescent):
    def __init__(self,size,RBsize,resTol):
        super().__init__()
        self._RBSize = RBsize
        self._needNewRB = True
        self._firstTime = True
        self._V = None
        self._snaps = np.zeros((size,self._RBSize))
        self._resTol = resTol
        self._nQR = 0
        self._nROMSolve = 0
        self._tQR = 0
        self._tROMSolve = 0

    def setRBSize(self,size):
        self._RBSize = size

    def GSBasis(self):
        t0 = time()
        Q,_ = np.linalg.qr(self._snaps)
        t1 = time()
        self._tQR += t1-t0
        self._nQR += 1
        return Q
    
    def solveState(self,x):
        U = None
        if self._it < self._RBSize:
            U = self._sol.compute(x)
            if self._state == 0:
                self._snaps[:,self._it] = U
        else:
            if self._needNewRB:
                if not self._firstTime:
                    U = self._sol.compute(x)
                    self._snaps[:,0:-1] = self._snaps[:,1:]
                    self._snaps[:,-1] = U
                self._V = self.GSBasis()
                self._needNewRB = False
                self._firstTime = False
            K = self._sol.getMatrix(x)
            F = self._sol.getRHS(x)

            t0 = time()
            Krb = self._V.T@K@self._V
            Frb = self._V.T@F
            Urb = np.linalg.solve(Krb,Frb)
            Uapp = self._V@Urb
            self._nROMSolve += 1
            t1 = time()
            self._tROMSolve += t1-t0
            freeDDLs = self._sol._problem._freeDDLs
            res = np.linalg.norm(K[freeDDLs][:,freeDDLs]@Uapp[freeDDLs]-F[freeDDLs])/np.linalg.norm(F[freeDDLs])
            if res > self._resTol:
                self._needNewRB = True
                if U is None:
                    U = Uapp.copy()
                print(f"Recompute reduced basis (res={res})")
            else:
                print(f"Reduced basis valid (res={res})")
                U = Uapp.copy()

        self._associatedAdjoint = -1*U.copy()
        return U

    def solveAdjoint(self,x):
        return self._associatedAdjoint
    


class GradientDescentPODEALS(GradientDescent):
    def __init__(self,size,RBsize,RBtol,resTol):
        super().__init__()
        self._RBSize = RBsize
        self._RBtol = RBtol
        self._V = None
        self._snaps = np.zeros((size,self._RBSize))
        self._resTol = resTol
        self._nSVD = 0
        self._nROMSolve = 0
        self._tSVD = 0
        self._tROMSolve = 0
        self._relresidue = np.Infinity
        self._needNewRB = False
        self._invalidRB = False


    def setRBSize(self,size):
        self._RBSize = size

    def acceptStep(self,x0,xt,fx0,fxt,p,gx0,s):
        suffDecr = super().acceptStep(x0,xt,fx0,fxt,p,gx0,s)
        validROM = self._relresidue < self._resTol if self._it >= self._RBSize else True
        # failure of linesearch
        if self._state == 2 and not self._invalidRB and self._lsit >= self._lsmax:
            self._needNewRB = True

        return suffDecr and validROM

    def PODBasis(self):
        nddl = self._snaps.shape[0]
        snaps_mean = np.reshape(np.mean(self._snaps,axis=1),(nddl,1))
        snaps_cov = (self._snaps-snaps_mean)
        t0 = time()
        VU, SU, _ = np.linalg.svd(snaps_cov, full_matrices=False)
        t1 = time()
        self._tSVD += t1-t0
        self._nSVD += 1
        i = 1
        ratio = 0
        while ratio < 1-self._RBtol:
            ratio = np.sum(SU[0:i])/np.sum(SU)
            i+=1
        print(f"Reduced basis size : {i} (energy ratio = {ratio})")
        return VU[:,0:i]
    
    def solveState(self,x):
        ddl = self._snaps.shape[0]
        U = None
        if self._it < self._RBSize:
            U = self._sol.compute(x)
            if self._state == 0:
                self._snaps[:,self._it] = U
        else:
            if (self._state == 0 and not self._invalidRB) or (self._needNewRB and self._state == 2):
                self._relresidue = np.Infinity
                U = self._sol.compute(x)
                if not np.allclose(U,self._snaps[:,-1]):
                    self._snaps[:,0:-1] = self._snaps[:,1:]
                    self._snaps[:,-1] = U
                    self._V = self.PODBasis()
                self._needNewRB = False
            else:
                K = self._sol.getMatrix(x)
                F = self._sol.getRHS(x)

                t0 = time()
                Krb = self._V.T@K@self._V
                snaps_mean = np.reshape(np.mean(self._snaps,axis=1),(ddl,1))
                Frb = self._V.T@F - (self._V.T@K@snaps_mean).flatten()
                Urb = np.linalg.solve(Krb,Frb)
                U = self._V@Urb + snaps_mean.flatten()
                self._nROMSolve += 1
                t1 = time()
                self._tROMSolve += t1-t0
                freeDDLs = self._sol._problem._freeDDLs
                prevres = self._relresidue
                self._relresidue = np.linalg.norm(K[freeDDLs][:,freeDDLs]@U[freeDDLs]-F[freeDDLs])/np.linalg.norm(F[freeDDLs])
                self._invalidRB = self._relresidue > self._resTol
                if self._invalidRB:
                    print(f"Reduced basis invalid (res={self._relresidue})")
                else:
                    print(f"Reduced basis valid (res={self._relresidue})")
                if self._state == 2 and self._invalidRB and self._lsit > 1:
                    self._needNewRB = np.abs(prevres-self._relresidue) < 1e-3
                if self._relresidue < 0.9*self._resTol and self._it >= self._RBSize and self._state==0:
                    self._increaseStep = True
                if self._it == self._RBSize:
                    self._backtrack = False

        self._associatedAdjoint = -1*U

        return U

    def solveAdjoint(self,x):
        return self._associatedAdjoint
