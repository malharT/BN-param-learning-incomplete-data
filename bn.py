from functools import reduce
import operator

import numpy as np

from factorset import factorset
from factor import discretefactor


# a Bayesian network is a factorset with
# added information mapping each variable to a factor
# (which is the CPD for that variable)
class bn(factorset):
    # structure should be a bnstructure (see below)
    # factors should be a factorset (see factorset.py)
    def __init__(self):
        super().__init__()
        self._findex = {} # maps variables to their factor index
        self._vindex = [] # maps factor indexes to the variable

    # v should be the variable for which f is the CPD
    def addfactor(self,f,v):
        i = super().addfactor(f)
        self._findex[v] = i
        self._vindex.append(v) #[i] = v
        return i

    def family(self,v):
        return self.factors[self._findex[v]].vars

    def parents(self,v):
        return self.family(v) - {v}

    def __str__(self):
        # needed because str() on a set uses repr(),
        #   not str() on underlying elements
        def settostr(s):
                return '{'+','.join([str(i) for i in s])+'}'

        ret = "variables: %s\n" % settostr(self.vars)
        for v in self.vars:
            ret += "%s: parents = %s\n" % (v,settostr(self.parents(v)))
            ret += str(self.factors[self._findex[v]])
            ret += "\n";
        return ret


class bnstructure:
    # vars should be a set of variables
    def __init__(self, vars):
        self._par = {v:set() for v in vars}

    def addvar(self,v):
        self._par[v] = set()

    # returns the set of parents of v
    def parents(self, v):
        return self._par[v] if v in self._par else set()

    # returns the set of parents of v
    def family(self, v):
        return self.parents(v).union({v})

    def addedge(self, fromv, tov):
        self._par[tov].add(fromv)

    def deledge(self, fromv, tov):
        self._par[tov].remove(fromv)

    @property
    def vars(self):
        return self._par.keys()

    # returns a bn in which every variable has a uniform
    #   distribution (irrespective of parent values) if v is None,
    #   or all factor values are "v" if v is not None
    # assumes discrete RVs!
    def randomBN(self):
        ret = bn()
        for v in self._par:
            fctr = discretefactor(self.family(v))
            fctr.phi = np.random.rand(*fctr.phi.shape)
            ret.addfactor(fctr, v)
        return ret
    def uniformBN(self,v=None):
        ret = bn()
        for v in self._par:
            if v is None:
                ret.addfactor(discretefactor(self.family(v),1.0/v.nvals),v)
            else:
                ret.addfactor(discretefactor(self.family(v),1.0/v.nvals),v)
        return ret

    @staticmethod
    def product_of_factors(factors):
        return reduce(operator.mul, factors)

    def prob_given_evidence(self, factors, evidence):
        reduced_factors = []
        for var in factors:
            fctr = factors[var]
            reduced_factor = fctr.reduce(evidence)
            reduced_factors.append(reduced_factor)

        reduced_prob_factor = self.product_of_factors(reduced_factors)
        return reduced_prob_factor

    def update_factors_given_assignment(
            self, parameters, factors, assigment, missing_vars):
        missing_vars_fctr = self.prob_given_evidence(
            parameters, assigment)
        denominator = missing_vars_fctr.marginalize(
            missing_vars)
        gradient_count = missing_vars_fctr/denominator

        for var in factors:
            fctr = factors[var]
            if fctr.vars.intersection(missing_vars):
                current_missing_vars = fctr.vars.intersection(missing_vars)
                vars_to_marginalize = missing_vars - current_missing_vars
                fctr[assigment] = (
                    fctr[assigment]
                    + gradient_count.marginalize(vars_to_marginalize).phi)
            else:
                fctr[assigment] = fctr[assigment] + 1

    def get_log_likelihood(self, learning, data_array, var_order):
        log_likelihood = -1
        for assignment in data_array:
            if np.any(np.isin(assignment, -1)):
                indexes_of_missing = np.where(assignment == -1)[0]
                missing_vars = set()
                for index in indexes_of_missing.flatten():
                    missing_vars.add(var_order[index])
                assignment_dict = {}
                for index in range(len(var_order)):
                    if assignment[index] != -1:
                        assignment_dict[var_order[index]] = assignment[index]
                if learning:
                    missing_vars_fctr = self.prob_given_evidence(
                        learning, assignment_dict)
                    denominator = missing_vars_fctr.marginalize(
                        missing_vars)
                    log_likelihood += np.log(denominator[assignment_dict])

            else:
                assignment_dict = {}
                for index in range(len(var_order)):
                    assignment_dict[var_order[index]] = assignment[index]
                if learning:
                    full_fctr = self.prob_given_evidence(
                        learning, assignment_dict)
                    log_likelihood += np.log(full_fctr[assignment_dict])
        return log_likelihood

    # returns a bn learned using maximum likelihood
    # from the dataset d
    # d is a vector of assignments
    #   (an assignment is a map from variables to values)
    # [this is not the most efficient representation of a dataset,
    #  but it will make the code simple]
    # assumes discrete RVs!
    # assumes complete data!
    def learn_params(self, data_file_name, var_order, learning_rate=0.0001, alg=None, data_array=None, starting_point=None):
        learned_factors_dict = {}
        liklihoods = []
        if starting_point is None:
            init_bn = self.randomBN()
            for var in self.vars:
                learned_factors_dict[var] = init_bn.factors[init_bn._findex[var]]
        else:
            learned_factors_dict = starting_point
        prev_loglikelihood = -1
        converged = False
        if data_array is None:
            data_array = np.genfromtxt(data_file_name, dtype=int, delimiter=', ')
        while not converged:
            factors = {}
            for var in self.vars:
                factors[var] = discretefactor(self.family(var))
            for assignment in data_array:
                if np.any(np.isin(assignment, -1)):
                    indexes_of_missing = np.where(assignment == -1)[0]
                    missing_vars = set()
                    for index in indexes_of_missing.flatten():
                        missing_vars.add(var_order[index])
                    assignment_dict = {}
                    for index in range(len(var_order)):
                        if assignment[index] != -1:
                            assignment_dict[var_order[index]] = assignment[index]
                    if learned_factors_dict:
                        self.update_factors_given_assignment(
                            learned_factors_dict, factors, assignment_dict,
                            missing_vars)

                else:
                    assignment_dict = {}
                    for index in range(len(var_order)):
                        assignment_dict[var_order[index]] = assignment[index]
                    for var in factors:
                        fctr = factors[var]
                        fctr[assignment_dict] = fctr[assignment_dict] + 1

            learning_buffer = {}
            if not learned_factors_dict:
                for var in factors:
                    factors[var] = factors[var]/factors[var].marginalize({var})
                    learned_factors_dict[var] = factors[var]
                    for var in learned_factors_dict:
                        num = learned_factors_dict[var]
                        den = learned_factors_dict[var].marginalize({var})
                        learned_factors_dict[var] = num/den
            elif alg == 'gd':
                for var in factors:
                    ratio = factors[var]-factors[var].marginalize({var})*learned_factors_dict[var]
                    ratio.phi = learning_rate * ratio.phi

                    param = np.log((learned_factors_dict[var]
                                    * learned_factors_dict[var].marginalize({var})).phi)
                    param = param + ratio.phi
                    learning_buffer[var] = discretefactor(learned_factors_dict[var].vars)
                    learning_buffer[var].vindex = learned_factors_dict[var].vindex
                    learning_buffer[var].phi = np.exp(param)
                    learning_buffer[var] = learning_buffer[var]/learning_buffer[var].marginalize({var})
            elif alg == 'em':
                for var in factors:
                    learning_buffer[var] = factors[var]/factors[var].marginalize({var})

            log_likelihood = self.get_log_likelihood(learning_buffer, data_array, var_order)
            if prev_loglikelihood != -1:
                if (log_likelihood - prev_loglikelihood)/prev_loglikelihood > 0.01:
                    learning_rate *= 0.1
                else:
                    learned_factors_dict = learning_buffer
                if -abs(log_likelihood - prev_loglikelihood)/prev_loglikelihood < 0.00001:
                    converged = True
            else:
                learned_factors_dict = learning_buffer
            prev_loglikelihood = log_likelihood
            liklihoods.append(log_likelihood)
        learned_bn = bn()
        for var in learned_factors_dict:
            learned_bn.addfactor(learned_factors_dict[var], var)
        return learned_bn, liklihoods