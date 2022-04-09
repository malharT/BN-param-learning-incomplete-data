import sys
import time

import numpy as np

from factor import discretevariable, discretefactor
from itertools import product

from bn import bnstructure
from bn import bn


def makefactor(vars, vals):
    phi = discretefactor(set(vars))
    for j,x in enumerate(product(*map((lambda v : [(v,i) for i in range(v.nvals)]),vars))):
        s = {a:b for (a,b) in x}
        phi[s] = vals[j]
    return phi

def buildnewstudentex():
    c = discretevariable("c",2)
    d = discretevariable("d",2)
    t = discretevariable("t",2)
    i = discretevariable("i",2)
    g = discretevariable("g",3)
    s = discretevariable("s",2)
    l = discretevariable("l",2)
    j = discretevariable("j",2)

    studentbn = bn()

    pc = makefactor([c],[0.5,0.5])
    studentbn.addfactor(pc,c)
    pd = makefactor([c,d],[0.4,0.6,0.8,0.2])
    studentbn.addfactor(pd,d)
    pi = makefactor([i],[0.6,0.4])
    studentbn.addfactor(pi,i)
    pt = makefactor([i,t],[0.9,0.1,0.4,0.6])
    studentbn.addfactor(pt,t)
    pg = makefactor([t,d,g],
        [0.3,0.4,0.3,
         0.05,0.25,0.7,
         0.9,0.08,0.02,
         0.5,0.3,0.2])
    studentbn.addfactor(pg,g)
    ps = makefactor([t,s],[0.95,0.05,0.2,0.8])
    studentbn.addfactor(ps,s)
    pl = makefactor([g,l],[0.1,0.9,0.4,0.6,0.99,0.01])
    studentbn.addfactor(pl,l)
    pj = makefactor([l,s,j],
        [0.9,0.1,
            0.4,0.6,
            0.3,0.7,
            0.1,0.9])
    studentbn.addfactor(pj,j)

    studentstr = bnstructure({c,d,t,i,g,s,l,j})
    studentstr.addedge(c,d)
    studentstr.addedge(i,t)
    studentstr.addedge(d,g)
    studentstr.addedge(t,g)
    studentstr.addedge(t,s)
    studentstr.addedge(s,j)
    studentstr.addedge(g,l)
    studentstr.addedge(l,j)

    return studentbn,(c,d,t,i,g,s,l,j),studentstr


def generate_data(studentbn, studentvars):
    sample_size = 100
    # change to a fixed seed to get consistent results
    np.random.seed(int(time.time()))
    file_data = ''
    for i in range(sample_size):
        sample = studentbn.sample()
        for var in studentvars:
            file_data += str(sample[var]) + ","
        file_data = file_data[:-2]
        file_data += '\n'
    file_data = file_data[:-1]
    data_file = open("studenbn" + str(sample_size) + "_data.csv", "w")
    data_file.write(file_data)
    data_file.close()


##  This is just a simple test that samples from the student
##  BN and then uses the samples to learn back the parameters
##  Your code will be tested on more strenuous examples
if __name__ == '__main__':
    studentbn, studentvars, studentstr = buildnewstudentex()
    # generate_data(studentbn, studentvars)
    for entries in ['1000']:
        for miss_prob in ['0.1']:
            for iteration in ['1']:
                data_file_name = 'studenbn'+ entries + '_' + miss_prob + '.csv'
                data_array = np.genfromtxt(data_file_name, dtype=int, delimiter=',')
                train_array = data_array[:int(0.8*data_array.shape[0])]
                test_array = data_array[int(0.8*data_array.shape[0]):]

                learnedbn1, likelihoods = studentstr.learn_params(
                    data_file_name, studentvars, learning_rate=0.001, alg='gd', data_array=train_array)
                print(len(likelihoods))
                exit()
                with open(data_file_name + "gd_lh_" + iteration + ".csv",'w') as lhf:
                    lhf.write('\n'.join([str(lh) for lh in likelihoods]))
                print("em starts")
                learnedbn2, likelihoods = studentstr.learn_params(
                    data_file_name, studentvars, alg='em', data_array=train_array)
                with open(data_file_name + "em_lh_" + iteration + ".csv",'w') as lhf:
                    lhf.write('\n'.join([str(lh) for lh in likelihoods]))
                gd_learning = {}
                for var in learnedbn1.vars:
                    gd_learning[var] = learnedbn1.factors[learnedbn1._findex[var]]
                em_learning = {}
                for var in learnedbn2.vars:
                    em_learning[var] = learnedbn2.factors[learnedbn2._findex[var]]
                gd_likelihood = studentstr.get_log_likelihood(gd_learning, test_array, studentvars)
                em_likelihood = studentstr.get_log_likelihood(em_learning, test_array, studentvars)
                print("gd likelihood: ", gd_likelihood)
                print("em likelihood: ", em_likelihood)
