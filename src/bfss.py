#binary FSS (domain is binary == sequence of 0s n 1s)
#problems being solved are discrete in nature (0/1 MKP belongs to this category)
#the fish class here has different operators(basically the displacement n feed works differently)

###INCOMPLETE


# vanilla FSS (this algorithm is used for continuous optimization problems)
# tests: find global maxima of Bivariate bell curve (unbounded == no constraints)
# chose Bell curves becoz they unimodular
# Population = school size = 30


import numpy as np
#import copy
from random import random,randint   
import time
import math



class School(object):
    def __init__(self, problem, population_size, dim, wscale, objtve, step_ind, thresh_c, thresh_v):
        self.problem = problem
        self.size = population_size
        self.w_scale = wscale
        self.dim = dim
        self.school = []
        self.prev_weight = self.w_scale * self.size
        self.curr_weight = self.w_scale * self.size
        self.step_ind = step_ind
        self.thresh_c = thresh_c
        self.thresh_v = thresh_v
        self.del_f_max = 0.0				#school fitness
        self.best_fish = None 			
        self.objective = objtve 			
        self.barycenter = np.zeros(self.dim ,dtype = float, order = 'C')
        self.col_ins_disp = np.zeros(self.dim ,dtype = float, order = 'C')


    def init_fish_school(self):
        self.school = [ Fish(self, generateRandBinSeq(self.dim, self.problem.constraints, self.problem.bounds) ) for _ in range(0,self.size)]
        self.update_best_fish()


    def print_school_info(self):
        print('School Weight = ', self.curr_weight)
        print('Prev School Weight = ', self.prev_weight)
        print('Maximum fitness gain = ', self.del_f_max)
        print('Bset Solution = ', self.best_fish.X)


    #basically what happens in 1 iteration
    #follows directly from the FSS algorithm pseudocode
    def update_school(self):
        #for each fish Perform the individual displacement (Equation 1)
        for i in self.school:
            i.displace_ind()
            #apply fitness function
            i.update_del_f()
            #i.print_fish_status()
        #update the best fish in school(according to current fitness)
        self.update_best_fish()
        #for each fish Update its weight
        for i in self.school:
            i.feed()
        #for each fish Perform the collective instinctive displacement
        #for that calculate the col ins disp vector == col ins disp
        self.update_col_ins_vec() # eqn 5
        for i in self.school:
            i.displace_col_ins() # x = x + m
        #for each fish Perform the collective volitive displacement
        #for that update school's barycenter
        self.update_barycenter()
        for i in self.school:
            i.displace_col_vol()
            i.update_del_f()
        self.update_best_fish()


    def update_col_ins_vec(self):
        sigma_del_f = 0.0
        for i in self.school:
            sigma_del_f += i.del_f
        self.col_ins_disp.fill(0.0)
        for i in self.school:
            self.col_ins_disp += i.del_f * (i.X - i.X_prev)
        if sigma_del_f:
            self.col_ins_disp *= 1/sigma_del_f
        else:
            self.col_ins_disp.fill(0.0)
        


    def update_best_fish(self):
        max = 0
        for i in range(0,self.size):
            if self.school[i].f > self.school[max].f:
                max = i
        self.best_fish = self.school[max] 
        print(self.best_fish)


    def update_barycenter(self):
        self.update_school_w()
        self.barycenter.fill(0.0)
        for i in self.school:
            self.barycenter += i.W * i.X
        self.barycenter *= 1/self.curr_weight
        #convert to binary coords
        max_b = np.amax(self.barycenter)
        for i in range(0,self.dim):
            if(self.barycenter[i] < self.thresh_v * max_b):
                self.barycenter[i] = 0
            else:
                self.barycenter[i] = 1
        


    def update_school_w(self):
        self.prev_weight = self.curr_weight
        self.curr_weight = 0.0
        for i in self.school:
            self.curr_weight += i.W
        
            

#step_ind = [0.0,1.0], thresh = [0.0,1.0)
class Fish:
    def __init__(self, school, x):
        self.school = school
        self.X = x
        self.X_prev = np.copy(self.X)
        self.W = self.school.w_scale/2
        self.del_f = 0.0
        self.f = getObjective(self.X, self.school.objective) # y = f(x)
        self.f_prev = getObjective(self.X, self.school.objective)

    def displace_ind(self):
        # try individual
        m = np.copy(self.X) # temp X
        for i in range(0,m.size):
            if(random() < self.school.step_ind):
                m[i] = not m[i] 
        np.copyto(self.X_prev, self.X, casting='same_kind', where=True)
        self.f_prev = self.f
        if check_constraints_linear(m, self.school.problem.constraints, self.school.problem.bounds) == False:
            return
        m_y = getObjective(m , self.school.objective)
        if m_y > self.f:
            self.f = m_y
            self.X = m

    def update_del_f(self):
        self.del_f = self.f - self.f_prev


    def feed(self):
        # weight can decrease if fitness decreases
        if(self.school.del_f_max):
           self.W += (self.del_f)/abs(self.school.del_f_max)
        self.W = min(self.W, self.school.w_scale)

    def displace_col_ins(self):
        max_m = np.amax(self.school.col_ins_disp)
        temp = np.copy(self.X) # temp X
        d = randint(0,self.school.dim - 1)
        if(self.school.col_ins_disp[d] < self.school.thresh_c * max_m):
            temp[d] = 0
        else:
            temp[d] = 1
        np.copyto(self.X_prev, self.X, casting='same_kind', where=True)
        self.f_prev = self.f
        if check_constraints_linear(temp, self.school.problem.constraints, self.school.problem.bounds) == False:
            return
        self.X = temp
        self.f = getObjective(self.X, self.school.objective)
        
        
        
    def displace_col_vol(self):
        d = randint(0,self.school.dim - 1)
        temp = np.copy(self.X)
        if(self.school.curr_weight - self.school.prev_weight > 0):
            temp[d] = self.school.barycenter[d]
        else:
            temp[d] = not self.school.barycenter[d]
        np.copyto(self.X_prev, self.X, casting='same_kind', where=True)
        self.f_prev = self.f
        if check_constraints_linear(temp, self.school.problem.constraints, self.school.problem.bounds) == False:
            return
        self.X = temp
        self.f = getObjective(self.X, self.school.objective)
            

    # debug functions
    def print_fish_status(self):
        print("X = ", self.X)
        print("X prev = ", self.X_prev)
        print("Weight = ", self.W)
        print("fitness = ", self.f)
        print("Prev fitness = ", self.f_prev)
        print("y = ", self.y)
        print("y prev = ", self.y_prev)

class Problem:
    def __init__(self, discrt, dim, minim, objtve, M, constrnt, bounds, optim=0, solved=False):
        self.discrete = discrt
        self.dim = dim
        self.minimize = minim
        self.objective = objtve
        self.n_constraint = M
        self.constraints = constrnt
        self.bounds = bounds
        self.optima = optim
        self.solved = solved
        
class Solver:
    def __init__(self, runs, iterations, problem, population_size, w_scale, step_ind, thresh_c, thresh_v):
        self.runs = runs
        self.T = iterations
        self.t = 0  #current interation
        self.problem = problem
        self.population = population_size
        self.w_scale = w_scale
        self.step_ind = step_ind
        self.thresh_c = thresh_c
        self.thresh_v = thresh_v
        self.school = School(self.problem, self.population, self.problem.dim, self.w_scale, self.problem.objective, self.step_ind, self.thresh_c, self.thresh_v)



# HELPER FUNCTIONS
def mutateBinSeq(dim, X):
    
    rand_start = randint(0,dim-1)
    for i in range(rand_start,dim):
        if(X[i] == 0):
            X[i] = 1
        else:
            X[i] = 0
    print('Mutation occured ', type(X))

def generateRandBinSeq(dim, constraints=None, bounds=None):
    # we need to change the dtype of X from int to float here
    X = np.zeros(dim, dtype=int, order='C')
    #print(X.size)
    for i in range(0, X.size):
        if(random() >= 0.5):
            X[i] = 1
    while(check_constraints_linear(X, constraints, bounds) == False):
        print('X before mutation ', X)
        mutateBinSeq(dim, X)
        print('X after mutation ', X)

    return X

def getObjective(X, objective):
    return X.dot(objective)         #assuming it will return their inner product


def get_euclidean_dist(n1, n2):
    n3 = n1 - n2
    return np.linalg.norm(n3)


def check_constraints_linear(X, coef, bounds):
    for i in range(0, len(coef)):
        linear_sum = X.dot(coef[i])
        if linear_sum > bounds[i]:
            return False
    return True


def main():
    obj = np.asarray([1,2,3], dtype=int)
    constr = [np.asarray([1,0,1]), np.asarray([0,1,1])]
    bound = [1,2]
    p = Problem(True, 3, False, obj, 2, constr, bound, 0, False)
    s = Solver(1, 5, p, 5, 30.0, 0.5, 0.4 ,0.4)
    s.school.init_fish_school()
    for c in range(0,100):
        print('iteration = ', c)
        print('best fish = ', s.school.best_fish.X, ' fitness = ', s.school.best_fish.f)
        for i in range(0,s.school.size):
            print(s.school.school[i].X,' ', check_constraints_linear(s.school.school[i].X, s.problem.constraints, s.problem.bounds))
            s.school.update_school()

    

    return 0



if __name__ == "__main__":
    main()

