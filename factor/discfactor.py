from itertools import count, product
import numpy as np
import operator
import copy
from .variable import *

class discretefactor:

    def __init__(self, varset, defval=0.0, vals=None):
        self.vars = varset
        self.vindex = {v:i for i,v in enumerate(varset)}
        if vals is None:
            self.phi = np.full(self.cards(),defval)
        else:
            if not isinstance(vals,np.ndarray):
                raise TypeError("discretefactor's initial values must be a numpy array")
            self.phi = vals # better be a np array of the right shape

    @property
    def scope(self):
        return self.vars

    def cards(self):
        return [x.nvals for x in self.vars]

    def npindextup(self, assign):
        # assign should be a dict mapping variables to values
        if not isinstance(assign,dict):
            raise TypeError("discretefactor lookup must be with a dictionary mapping variables to values")
        return tuple((assign[x] if x in assign else slice(None)) for x in self.vars)

    def __getitem__(self, assign):
        return self.phi[self.npindextup(assign)]

    def __setitem__(self, assign, value):
        self.phi[self.npindextup(assign)] = value

    def __str__(self):
        ret = "";
        for v in self.vars:
            ret += str(v) + " "
        ret += "\n"
        for x in product(*[range(x) for x in self.cards()]):
            ret += reduce(operator.add, (str(v)+" " for v in x),"")
            ret += " " + str(self.phi[x]) + "\n"
        return ret

    def reorgphi(self, vs):
        exindnum = count(len(self.phi.shape))
        newind = [(self.vindex[v] if v in self.vars else next(exindnum)) for v in vs]
        nadd = next(exindnum)-len(self.phi.shape)
        return np.transpose(np.reshape(self.phi,self.phi.shape+((1,)*nadd)),newind)

    def remap(self, toremove, values):
        newscope = {v for v in self.vars if v not in toremove}
        nnum = count(0)
        newindmap = [(next(nnum) if v not in toremove else 0) for v in self.vars]
        newind = [newindmap[self.vindex[v]] for v in newscope]
        return discretefactor(newscope,values.transpose(newind))

    def reduce(self, assign):
        return self.remap(assign,self.phi[self.npindextup(assign)])

    def marginalize(self, tosumout): 
        # tosumout is a set of variables
        return self.remap(tosumout,
            np.sum(self.phi,axis=
                tuple(i for i,v in enumerate(self.vars) if v in tosumout)))

    def dobinop(self,other,binop):
        ttlvars = self.vars.union(other.vars)
        return discretefactor(ttlvars,
                binop(self.reorgphi(ttlvars),other.reorgphi(ttlvars)))

    def __add__(self,other):
        return self.dobinop(other,np.add)

    def __sub__(self,other):
        return self.dobinop(other,np.subtract)

    def __mul__(self,other):
        return self.dobinop(other,np.multiply)

    def __truediv__(self,other):
        return self.dobinop(other,np.divide)

    def maxabs(self):
        return np.max(np.abs(self.phi))
