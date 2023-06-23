import numpy as np
import matplotlib.pyplot as plt
import abc

######################################################
#
# MESH 
#
######################################################

class Mesh:
    _nelem = 0
    _nnodes = 0
    _nddls = 0
    _nddls = 0
    _coord = None
    _connec = None
    _numerVec = None
    _addresVec = None
    _numerScal = None
    _addresScal = None
    _centr = None
    _top = None
    _bottom = None 
    _left = None 
    _right = None

    def __init__(self, nelemx, nelemy):
        self._nnodes = (2*nelemx+1)*(2*nelemy+1)
        self._nelem = 2*nelemx*nelemy
        self._nddls = 2*self._nnodes
        self._generateMesh(nelemx, nelemy)

    def getNnodes(self):
        return self._nnodes
    
    def getNddlsScal(self):
        return int(self._nddls/2)

    def getNddlsVec(self):
        return self._nddls
    
    def getNelems(self):
        return self._nelem

    def getElemNodes(self):
        return self._connec.copy()

    def getNodeCoord(self):
        return self._coord.copy()
    
    def getElemDDLScal(self):
        return self._addresScal.copy()

    def getDDLnumerScal(self):
        return self._numerScal.copy()
    
    def getElemDDLVec(self):
        return self._addresVec.copy()

    def getDDLnumerVec(self):
        return self._numerVec.copy()
        
    def getElemCentr(self):
        return self._centr.copy()

    def getTopBound(self):
        return self._top.copy()

    def getBottomBound(self):
        return self._bottom.copy()
    
    def getLeftBound(self):
        return self._left.copy()
    
    def getRightBound(self):
        return self._right.copy()

    def _generateMesh(self, nelemx, nelemy):
        ## CONSTRUCTION DU MAILLAGE
        # Géométrie rectangulaire HxL
        coin = [0,0]
        H = 1
        L = 2

        # Création du maillage (il y a 2 x nelemx x nelemy triangles)
        meshx,meshy = np.meshgrid(np.linspace(coin[0], coin[0]+L, 2*nelemx+1), np.linspace(coin[1], coin[1]+H, 2*nelemy+1))

        # Numérotation globale des noeuds
        self._coord = np.vstack((meshx.flatten(), meshy.flatten())).T

        # Tableau de connectivité des éléments : elem -> num. des noeuds
        # Éléments pairs (3 premiers sont les noeuds géo)
        # connec = [i, i+2, i+4*nelemx+2, i+1, i+2*nelemx+1, i+2*nelemx+2];
        # Éléments impairs
        # connec = [(i+1)+4*nelemx+2, (i+1)+4*nelemx, i+1, (i+1)+4*nelemx+1, (i+1)+2*nelemx+1, (i+1)+2*nelemx]
        evenelems = np.array(range(0,self._nelem,2))
        oddelems = np.array(range(1,self._nelem,2))
        evenshift = (2*nelemx+2)*(evenelems//(2*nelemx))
        oddshift = (2*nelemx+2)*(oddelems//(2*nelemx))

        connecP1 = np.zeros((self._nelem,3)) # connec P1 contient seulement les noeuds géométriques
        connecP1[evenelems,:] = np.vstack((evenelems + evenshift,evenelems+2+evenshift,evenelems+4*nelemx+2+evenshift)).T
        connecP1[oddelems,:] = np.vstack(((oddelems+1)+4*nelemx+2+oddshift,(oddelems+1)+4*nelemx+oddshift,oddelems+1+oddshift)).T
        connecP1 = connecP1.astype("int")

        self._connec = np.zeros((self._nelem,6))
        self._connec[:,:3] = connecP1
        self._connec[evenelems,3:] = np.vstack((evenelems+1+ evenshift,evenelems+2*nelemx+1+ evenshift,evenelems+2*nelemx+2+ evenshift)).T
        self._connec[oddelems,3:] = np.vstack(((oddelems+1)+4*nelemx+1+oddshift,(oddelems+1)+2*nelemx+1+oddshift,(oddelems+1)+2*nelemx+oddshift)).T
        self._connec = self._connec.astype("int")

        # Centroides des éléments
        self._centr = np.sum(self._coord[connecP1],axis=1)/3

        ## NUMÉROTATION DES DEGRÉS DE LIBERTÉ (noeud i -> DDLs (2i, 2i+1) pour vec, sinon i -> i en scal)
        self._numerScal = np.arange(self._nnodes).astype("int")
        self._addresScal = self._connec.copy()

        self._numerVec = np.vstack((2*np.arange(self._nnodes),2*np.arange(self._nnodes)+1)).T.astype("int")
        self._addresVec = np.zeros((self._nelem,12))
        self._addresVec[:,:6] = 2*self._connec
        self._addresVec[:,6:] = 2*self._connec+1
        self._addresVec = self._addresVec.astype("int")

        # Détermination des bords
        self._left = np.arange(0,self._nelem,2*nelemx)
        self._right = np.arange(2*nelemx-1,self._nelem,2*nelemx)
        self._bottom = np.arange(0,2*nelemx-1,2)
        self._top = np.arange(2*(nelemy-1)*nelemx+1,self._nelem,2)

    def displayMesh(self):
        plt.figure(figsize=(16,8))
        plt.scatter(self._coord[:,0],self._coord[:,1])
        for i, txt in enumerate(range(self._nnodes)):
            plt.annotate(txt, (self._coord[i,0], self._coord[i,1]))
        plt.scatter(self._centr[:,0],self._centr[:,1],marker="x")
        plt.legend(["Nodes","Elements (centr.)"])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def whichBoundaryIsElemK(self, k):
        which = None
        if k in self._left:
            which = 0
        elif k in self._bottom:
            which = 1
        elif k in self._right:
            which = 2
        elif k in self._top:
            which = 3
        return which

######################################################
#
# TRANSFORMATION TO REFERENCE ELEMENT/INTEGRATION
#
######################################################

class RefElement:

    _Tb = None
    _gradf = np.zeros((4,2,6))
    _f = np.zeros((4,6))
    _mesh = None

    # Points de Hammer
    xi = [1/3, 1/5, 1/5, 3/5]
    yi = [1/3, 1/5, 3/5, 1/5]
    wi = [-0.281250, 0.2604166666666667, 0.2604166666666667, 0.2604166666666667]

    # Fonctions de Lagrange quad. sur l'élément de référence
    f0 = lambda x : 2*x[0]**2 + 4*x[0]*x[1] - 3*x[0] + 2*x[1]**2 - 3*x[1] + 1
    df0dx = lambda x : 4*x[0] + 4*x[1] - 3
    df0dy = df0dx
    f1 = lambda x : x[0]*(2*x[0]-1)
    df1dx = lambda x : 4*x[0] - 1
    df1dy = lambda x : 0
    f2 = lambda x : x[1]*(2*x[1]-1)
    df2dx = lambda x : 0
    df2dy = lambda x : 4*x[1] - 1
    f3 = lambda x : 4*x[0]*(-x[0]-x[1]+1)
    df3dx = lambda x : -8*x[0] - 4*x[1] + 4
    df3dy = lambda x : -4*x[0]
    f4 = lambda x : 4*x[1]*(-x[0]-x[1]+1)
    df4dx = lambda x : -4*x[1]
    df4dy = lambda x : -8*x[1] -4*x[0] + 4
    f5 = lambda x : 4*x[0]*x[1]
    df5dx = lambda x : 4*x[1]
    df5dy = lambda x : 4*x[0]

    fi = lambda x : np.array([RefElement.f0(x), RefElement.f1(x), RefElement.f2(x), RefElement.f3(x), RefElement.f4(x), RefElement.f5(x)])
    deriv = lambda x : np.array([[RefElement.df0dx(x), RefElement.df1dx(x), RefElement.df2dx(x), RefElement.df3dx(x), RefElement.df4dx(x), RefElement.df5dx(x)],[RefElement.df0dy(x), RefElement.df1dy(x), RefElement.df2dy(x), RefElement.df3dy(x), RefElement.df4dy(x), RefElement.df5dy(x)]])

    def __init__(self, mesh):
        self._mesh = mesh
        self._computeTransformations(mesh)
        self._evalBasisFuncAndGradAtQuadPts()

    def _computeTransformations(self, mesh : Mesh):
        self._Tb = np.zeros((2*mesh.getNelems(),3))
        for i in range(mesh.getNelems()):
            vert = mesh.getNodeCoord()[mesh.getElemNodes()[i,:3],:]
            bk = vert[0,:].T
            self._Tb[2*i:2*(i+1),2] = bk
            self._Tb[2*i:2*(i+1),:2] = vert[1:,:].T-np.repeat(bk.reshape((2,1)),2,1)

    def _evalBasisFuncAndGradAtQuadPts(self):
        for i in range(4):
            self._gradf[i,:,:] = RefElement.deriv([RefElement.xi[i],RefElement.yi[i]])
            self._f[i,:] = RefElement.fi([RefElement.xi[i],RefElement.yi[i]])

    def transformPointsToElemK(self, pts, k):
        TK = self._Tb[2*k:2*(k+1),:2]
        bK = self._Tb[2*k:2*(k+1),2].reshape((2,1))
        return (TK@pts.T + bK).T

    def evalfi(self, pts, i):
        return np.array([RefElement.fi([pts[j,0],pts[j,1]])[i] for j in range(pts.shape[0])])

    def intM1gradfikDotM2gradfjk(self,M1,M2,i,j,k):
        TK = self._Tb[2*k:2*(k+1),:2]
        invDTK = np.linalg.inv(TK).T
        JK = np.linalg.det(TK)
        val = 0
        for s in range(4):
            val += RefElement.wi[s]*(M2@invDTK@self._gradf[s,:,j].reshape((2,1))).T@M1@invDTK@self._gradf[s,:,i].reshape((2,1))
        return JK*val
        
    def intgxfik(self,g,i,k):
        TK = self._Tb[2*k:2*(k+1),:2]
        JK = np.linalg.det(TK)
        bK = self._Tb[2*k:2*(k+1),2].reshape((2,1))
        val = 0
        for s in range(4):
            p = np.array([[RefElement.xi[s]],[RefElement.yi[s]]])
            val += RefElement.wi[s]*g(TK@p+bK)*self._f[s,i]
        return JK*val
    
    def intYfik(self,i):
        # Some assumptions can be done because of the structured mesh :
        # 1) Integration will only ever occur on faces aligned with y axis
        # 2) The ratio of length deformation is 1/nelemy
        # 3) Basis functions are deg. 2 so 2 points Gauss quad. is sufficient
        return 0.5*(RefElement.fi([0,(np.sqrt(3)-1)/(2*np.sqrt(3))])[i]+RefElement.fi([0,(np.sqrt(3)+1)/(2*np.sqrt(3))])[i])/self._mesh.getLeftBound().shape[0]

    def intXfik(self,i):
        # Some assumptions can be done because of the structured mesh :
        # 1) Integration will only ever occur on faces aligned with x axis
        # 2) The ratio of length deformation is 2/nelemx
        # 3) Basis functions are deg. 2 so 2 points Gauss quad. is sufficient
        return 0.5*(RefElement.fi([(np.sqrt(3)-1)/(2*np.sqrt(3)),0])[i]+RefElement.fi([(np.sqrt(3)+1)/(2*np.sqrt(3)),0])[i])*(2/self._mesh.getTopBound().shape[0])

    def displaySolution(self,U,gridsize,dim):
        assert(dim==1 or dim==2)
        addres = self._mesh.getElemDDLScal() if dim==1 else self._mesh.getElemDDLVec()
        nelem = self._mesh.getNelems()
        # Define points on the reference triangle
        inter = np.linspace(0,1,gridsize)
        pts = []
        for i in range(inter.shape[0]):
            for j in range(inter.shape[0]-i):
                pts.append([inter[i],inter[j]])
        pts = np.array(pts)
        npts = pts.shape[0]

        vizmag = np.zeros((nelem*npts,5))
        for k in range(nelem):
            Tpts = self.transformPointsToElemK(pts,k)
            valx = np.zeros(npts)
            valy = np.zeros(npts)
            for j in range(6):
                ujkx = U[addres[k,j]]
                ujky = U[addres[k,j+6]] if dim==2 else 0
                fj = self.evalfi(pts,j)
                valx += ujkx*fj
                valy += ujky*fj
            vizmag[npts*k:npts*(k+1),0] = Tpts[:,0]
            vizmag[npts*k:npts*(k+1),1] = Tpts[:,1]
            vizmag[npts*k:npts*(k+1),2] = np.sqrt(valx**2 + valy**2)
            vizmag[npts*k:npts*(k+1),3] = valx
            vizmag[npts*k:npts*(k+1),4] = valy
        # x disp.
        plt.figure(figsize=(12,6))
        plt.set_cmap("viridis")
        plt.scatter(vizmag[:,0],vizmag[:,1],c=vizmag[:,3],marker="s")
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("u1")
        plt.show()
        if dim == 2:
            # y disp.
            plt.figure(figsize=(12,6))
            plt.set_cmap("viridis")
            plt.scatter(vizmag[:,0],vizmag[:,1],c=vizmag[:,4],marker="s")
            plt.colorbar()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("u2")
            plt.show()
            # disp. mag.
            plt.figure(figsize=(12,6))
            plt.set_cmap("viridis")
            plt.scatter(vizmag[:,0],vizmag[:,1],c=vizmag[:,2],marker="s")
            plt.colorbar()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("||u||")
            plt.show()


######################################################
#
# WEAK FORMULATION
#
######################################################
class IntegralTerm:

    _refElem = None
    _nElem = 0

    def __init__(self, refelem, nelem):
        self._refElem = refelem
        self._nElem = nelem

    @abc.abstractmethod
    def integrate(self, i, j, k):
        return None

class RHSIntegralTerm:

    _refElem = None

    def __init__(self, refelem):
        self._refElem = refelem

    @abc.abstractmethod
    def integrate(self, i, k):
        return None

class DiffusionTerm(IntegralTerm):
    _tensor = None

    def __init__(self, refelem, nelem):
        super().__init__(refelem, nelem)

    def setParams(self,tensor):
        self._tensor = tensor

    def integrate(self, i, j, k):
        return self._refElem.intM1gradfikDotM2gradfjk(self._tensor,np.eye(2),i%6,j%6,k)


class ElasticityTerm(IntegralTerm):
    _YoungMod = 0
    _Nu = 0

    def __init__(self, refelem, nelem):
        super().__init__(refelem, nelem)

    def setParams(self,E,nu):
        self._YoungMod = E*np.ones(self._nElem)
        self._Nu = nu

    def integrate(self, i, j, k):
        # Propriétés mécaniques (HPP + plane stress, Voigt form)
        # Champ du module de Young défini par élément, les DDLs correspondent aux numéros d'éléments
        nu = self._Nu
        C = (1/(1-nu**2))*np.array([[1,nu,0],[nu,1,0],[0,0,0.5*(1-nu)]])
        Eps = lambda i : np.array([[1,0],[0,0],[0,1]]) if i<6 else np.array([[0,0],[0,1],[1,0]])
        return self._refElem.intM1gradfikDotM2gradfjk((self._YoungMod[k]*C)@Eps(i),Eps(j),i%6,j%6,k)

class SourceTermScal(RHSIntegralTerm):
    _Fb = None

    def __init__(self, refelem):
        super().__init__(refelem)

    def setParams(self,Fb):
        self._Fb = Fb

    def integrate(self, i, k):
        return self._refElem.intgxfik(self._Fb,i%6,k)

class SourceTermVec(RHSIntegralTerm):
    _Fb = None

    def __init__(self, refelem):
        super().__init__(refelem)

    def setParams(self,Fb):
        self._Fb = Fb

    def integrate(self, i, k):
        ei = lambda i : np.array([1,0]) if i<6 else np.array([0,1])
        g = lambda x : np.dot(self._Fb(x),ei(i))
        return self._refElem.intgxfik(g,i%6,k)

######################################################
#
# PROBLEM/SOLUTION
#
######################################################

class Problem:
    _mesh = None
    _refElem = None 
    _currentSol = None
    _currentMatrix = None
    _currentRHS = None
    _freeDDLs = None
    _dirichletNodal = None
    _neumannNodal = None
    _currentElemMat = None
    _nddls = None
    _addres = None
    _dim = 0
    _MatrixContribution = []
    _RHSContribution = []

    # TODO: more granular resolution (separate assembly phases)

    def __init__(self, mesh, dim):
        assert(dim==1 or dim==2)
        self._mesh = mesh
        self._refElem = RefElement(mesh)
        self._dim = dim
        if dim == 1:
            self._nddls = mesh.getNddlsScal()
            self._addres = mesh.getElemDDLScal()
        else:
            self._nddls = mesh.getNddlsVec()
            self._addres = mesh.getElemDDLVec()
        self._dirichletNodal = np.zeros(self._nddls)
        self._neumannNodal = np.zeros(self._nddls)

    def addTerm(self, term):
        self._MatrixContribution.append(term)

    def addTermRHS(self, term):
        self._RHSContribution.append(term)

    def setDirichletBC(self, noDDLs, valDDLs):
        self._freeDDLs = np.ones(self._nddls)
        self._freeDDLs[noDDLs] = 0
        self._freeDDLs = self._freeDDLs.astype("bool")
        self._dirichletNodal[noDDLs] = valDDLs

    def setNeumannBC(self, noDDLs, valDDLs):
        for i in range(noDDLs.shape[0]):
            pos = np.argwhere(self._addres == noDDLs[i])
            nNeighb = pos.shape[0]
            side = set({})
            for k in pos[:,0]:
                s = self._mesh.whichBoundaryIsElemK(k)
                if s is not None:
                    side.add(s)
            if len(side)==1: # This will filter out corners and interior nodes
                # Because of the structure of the mesh, it reduces to 4 cases
                if nNeighb == 1:
                    if 0 in side or 2 in side:
                        self._neumannNodal[noDDLs[i]] = valDDLs[i]*self._refElem.intYfik(4)
                    elif 1 in side or 3 in side:
                        self._neumannNodal[noDDLs[i]] = valDDLs[i]*self._refElem.intXfik(3)
                else:
                    if 0 in side or 2 in side:
                        self._neumannNodal[noDDLs[i]] = valDDLs[i]*(self._refElem.intYfik(0)+self._refElem.intYfik(2))
                    elif 1 in side or 3 in side:
                        self._neumannNodal[noDDLs[i]] = valDDLs[i]*(self._refElem.intXfik(0)+self._refElem.intXfik(1))

    def solve(self):
        ## ASSEMBLAGE DU SYSTÈME GLOBAL
        nelem = self._mesh.getNelems()

        A = np.zeros((self._nddls,self._nddls))
        F = np.zeros(self._nddls)
        Aelem = np.zeros((self._nddls,nelem*self._nddls))
        for k in range(nelem):
            for i in range(self._addres.shape[1]):
                ddli = self._addres[k,i]
                for Int in self._RHSContribution:
                    F[ddli] += Int.integrate(i,k)
                for j in range(self._addres.shape[1]):
                    ddlj = self._addres[k,j]
                    for Int in self._MatrixContribution:
                        valint = Int.integrate(i,j,k)
                        A[ddli,ddlj] += valint
                        F[ddli] -= self._dirichletNodal[ddlj]*valint
                    Aelem[ddli,ddlj+k*self._nddls] += A[ddli,ddlj]     
        
        # Resolution
        U = np.linalg.solve(A[self._freeDDLs][:,self._freeDDLs],F[self._freeDDLs]+self._neumannNodal[self._freeDDLs])
        self._currentSol = np.zeros(self._nddls)
        self._currentSol[self._freeDDLs] = U
        self._currentSol += self._dirichletNodal
        self._currentMatrix = A
        self._currentRHS = F
        self._currentElemMat = Aelem
        return self._currentSol.copy()

    def getCurrentMatrix(self):
        return self._currentMatrix

    def getCurrentElemMat(self):
        return self._currentElemMat
        
    def getCurrentRHS(self):
        return self._currentRHS

    def getCurrentSolution(self):
        return self._currentSol