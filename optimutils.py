import numpy as np


def DescentMethod(x0, gtol, steptol, nmax, f, grad, direction=None, lb=np.nan, ub=np.nan, s0=1, armijo=None):
    ngrad = np.Infinity
    ngradp = np.Infinity
    step = np.Infinity
    s = s0
    i = 0

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

    while ngrad > gtol and ngradp > gtol and step > steptol and i < nmax:

        # Precompute known quantities at current iterate
        fx0 = f(x0)
        gx0 = grad(x0)
        ngrad = np.sqrt(np.sum(gx0**2))

        # Compute projected descent direction
        p = direction(x0,gx0) if direction is not None else -1*gx0
        p = project_direction(x0,p)
        ngradp = np.linalg.norm(p)
        #p = p/np

        # Compute next iterate
        xt = project_point(x0 + s0*p)
        fxt = f(xt)

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
                fxt = f(xt)

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

def DescentMethod(x0, gtol, steptol, nmax, f, grad, direction=None, lb=np.nan, ub=np.nan, s0=1, armijo=None):
    ngrad = np.Infinity
    ngradp = np.Infinity
    step = np.Infinity
    s = s0
    i = 0

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

    while ngrad > gtol and ngradp > gtol and step > steptol and i < nmax:

        # Precompute known quantities at current iterate
        fx0 = f(x0)
        gx0 = grad(x0)
        ngrad = np.sqrt(np.sum(gx0**2))

        # Compute projected descent direction
        p = direction(x0,gx0) if direction is not None else -1*gx0
        p = project_direction(x0,p)
        ngradp = np.linalg.norm(p)
        #p = p/np

        # Compute next iterate
        xt = project_point(x0 + s0*p)
        fxt = f(xt)

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
                fxt = f(xt)

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