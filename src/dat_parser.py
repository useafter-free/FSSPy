"""
Will parse .dat dataset files
and initialize input variables like X and constraints list(the 2d constraint matrix) and other related variables




UNDER CONSTRUCTION-------------><
"""
import numpy as np
from Problem import Problem

def parse_single_instance(f_path):
    with open(f_path, 'r') as f:
        flat_list=[word for line in f for word in line.split()]
    numbers_parsed = []
    for word in flat_list:
        numbers_parsed.append(int(word))
    dim = int(numbers_parsed[0])
    M = int(numbers_parsed[1])
    objective = []
    for i in range(0,dim):
        objective.append(numbers_parsed[3+i])
    objective = np.asarray(objective)
    constraints = []
    for i in range(1,M + 1):
        cons = []
        for j in range(0,dim):
            cons.append(numbers_parsed[3+i*dim+j])
        constraints.append(np.asarray(cons))
    bounds = []
    for i in range(0,M):
        bounds.append(numbers_parsed[3+(M+1)*dim+i])
    optima = numbers_parsed[2]
    p = Problem(True, dim, False, objective, M, constraints, bounds, optima)
    return p
    # print('Objective')
    # print(objective)
    # print('Constraints')
    # print(constraints)
    # print('Bounds')
    # print(bounds)
    



def parse_multiple_instances():
    pass


def main():
    f_path = '../test/datasets/MKP/chubeas/OR30x100/OR30x100-0.25_10.dat'
    print('Dimension of X = ', N)
    print('No. of constraints = ', M)
    if(optima != 0):
        print('Optimal Solution = ', optima)
if __name__ == "__main__":
    main()

