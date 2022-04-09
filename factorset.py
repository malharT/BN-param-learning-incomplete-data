# construct with factorset()
# can only add (not remove) factors, currently
# use fs.vars to get the set of variables mentioned in the factors
# use fs.factors to get an array of the factors

class factorset:
    def __init__(self):
        self._factors = []
        self._vars = set()

    def addfactor(self,f):
        i = len(self._factors)
        self._factors.append(f)
        self._vars = self._vars.union(f.vars)
        return i

    @property
    def vars(self):
        return self._vars

    @property
    def factors(self):
        return self._factors
