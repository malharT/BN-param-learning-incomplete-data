import numpy as np

# supplied, in case you want to compare the results
# of different computations
def samefactor(f1,f2,eps=1e-6):
    if type(f1) != type(f2):
        return False
    if f1.scope != f2.scope:
        return False
    phi1 = f1.phi
    newind = [f2.vindex[v] for v in f1.vars]
    phi2 = f2.phi.transpose(newind)
    diffval = np.amax(np.abs(phi1-phi2))
    if phi1.shape != phi2.shape:
        return False

    return diffval<eps
