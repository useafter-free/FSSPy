"""
Will parse .dat dataset files
and initialize input variables like X and constraints list(the 2d constraint matrix) and other related variables




UNDER CONSTRUCTION-------------><
"""
import numpy as np



def main():
    f_path = '../test/datasets/MKP/chubeas/OR30x100/OR30x100-0.25_10.dat'
    with open(f_path, 'r') as f:
        flat_list=[word for line in f for word in line.split()]
    numbers_parsed = []
    for word in flat_list:
        numbers_parsed.append(int(word))
    N = numbers_parsed[0]
    M = numbers_parsed[1]
    constraints = []
    for i in range(0,M):
        cons = []
        for j in range(0,N):
            cons.append(numbers_parsed[3+i*N+j])
        constraints.append(cons)
    for c in constraints:
        print(c)


    optima = numbers_parsed[2]
    print('Dimension of X = ', N)
    print('No. of constraints = ', M)
    if(optima != 0):
        print('Optimal Solution = ', optima)
if __name__ == "__main__":
    main()

