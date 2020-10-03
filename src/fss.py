#!/usr/bin/env python


# vanilla FSS (this algorithm is used for continuous optimization problems)
# tests: find global maxima of Bivariate bell curve (unbounded == no constraints)
# chose Bell curves becoz they unimodular
# Population = school size = 30


import numpy as np
#import copy
import random
import math
# import check_constraints as strain


class School(object):
    def __init__(self, population_size, dim, wscale, fun, stepi, stepvol):
        self.size = population_size
        self.w_scale = wscale
        self.dim = dim
        self.school = []
        self.prev_weight = self.w_scale * self.size
        self.curr_weight = self.w_scale * self.size
        self.step_ind = stepi
        self.step_vol = stepvol
        self.del_f_max = 0.0				#school fitness
        self.best_fish = None 			#fish with max Weight(index of that fish) or an iterator
        self.objective = fun 			#will be a function ptr or lambda
        self.barycenter = np.zeros(self.dim ,dtype = float, order = 'C')
        self.col_ins_disp = np.zeros(self.dim ,dtype = float, order = 'C')


    def init_fish_school(self):
        self.school = [(Fish(self.dim, generateRandList(self.dim, -30,31), self.w_scale/2, testFunction1)) for _ in range(self.size)]


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
            i.displace_ind(self.step_ind)
            #apply fitness function
            i.update_fitness()
            #i.print_fish_status()
        #update the best fish in school(according to current fitness)
        self.update_best_fish()
        #for each fish Update its weight
        for i in self.school:
            i.feed(self.del_f_max, self.w_scale)
        #for each fish Perform the collective instinctive displacement
        #for that calculate the col ins disp vector == col ins disp
        self.update_col_ins_vec()
        for i in self.school:
            i.displace_col_ins(self.col_ins_disp)
        #for each fish Perform the collective volitive displacement
        #for that update school's barycenter
        self.update_barycenter()
        if self.curr_weight > self.prev_weight:
            self.step_vol = abs(self.step_vol)
        else:
            self.step_vol = -1 * abs(self.step_vol)
        for i in self.school:
            i.displace_col_vol(self.barycenter, self.step_vol)
            i.update_fitness()
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
        


    def update_del_f_max(self):
        max = self.school[0]
        for i in self.school:
            if self.school[i].del_f > max.del_f:
                max = self.school[i]
        self.del_f_max = max.del_f


    def update_best_fish(self):
        max = self.school[0]
        for i in self.school:
            if i.f > max.f:
                max = i
        self.best_fish = max


    def update_barycenter(self):
        self.update_school_w()
        self.barycenter.fill(0.0)
        for i in self.school:
            self.barycenter += i.W * i.X
        self.barycenter *= 1/self.curr_weight
        print(self.barycenter)


    def update_school_w(self):
        self.prev_weight = self.curr_weight
        self.curr_weight = 0.0
        for i in self.school:
            self.curr_weight += i.W
        print(self.curr_weight)
        
            

#import check_constraints as strain
class Fish:
    def __init__(self, dim, x, w, fun):
        self.dim = dim
        self.X = x
        self.X_prev = np.copy(self.X)
        self.W = w
        self.f = 1.0
        self.f_prev = 1.0
        self.del_f = 0.0
        self.objective = fun
        self.y = self.objective(self.X)
        self.y_prev = self.objective(self.X_prev)

    def displace_ind(self, step_ind):
        m = np.copy(self.X)
        with np.nditer(m, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = x + step_ind * random.uniform(-1.0, 1.0)
        m_y = self.objective(m)
        if m_y > self.y:
            self.y_prev = self.y
            self.y = m_y
            np.copyto(self.X_prev, self.X, casting='same_kind', where=True)
            self.X = m
        else:
            self.y_prev = self.y

    def update_fitness(self):
        self.f_prev = self.f
        self.f += (self.y - self.y_prev)
        self.del_f = self.f - self.f_prev


    def feed(self, del_f_max, w_scale):
        # weight can decrease if fitness decreases
        if(del_f_max):
           self.W += (self.f - self.f_prev)/abs(del_f_max)
        self.W = min(self.W, w_scale)

    def displace_col_ins(self, m):
        #m = np.zeroes(self.dim, dtype=float, order='C')
        # is it just holding a reference (beocz then i will have to copy it everytime)
        self.y_prev = self.y
        self.X += m  # numpy arr can be added like matrices ryt?
        

    def displace_col_vol(self, bary, step_vol):
        distance = get_euclidean_dist(self.X, bary)
        # again idk if we can scale numpy array like matrices
        self.X += random.uniform(0,1) * (step_vol / distance) * (self.X- bary)
        self.y = self.objective(self.X)

    # debug functions
    def print_fish_status(self):
        print("X = ", self.X)
        print("X prev = ", self.X_prev)
        print("Weight = ", self.W)
        print("fitness = ", self.f)
        print("Prev fitness = ", self.f_prev)
        print("y = ", self.y)
        print("y prev = ", self.y_prev)




"""
# for now this class wont be used much
class Problem:
	# def __init__(self, dim, size, dataset):
	# 	self.dim = dim
	# 	self.size = size
	# 	self.dataset = dataset
	# 	self.constraints = [[0 for _ in range(self.dim)] for _ in range(self.dim)]
	def __init__(self, dim, objective, constraints, opt):
		self.dim = dim
		self.obj_f = objective				# a function
		self.constraints = constraints 		# None if unbounded	
		self.opt_t = opt 					# minimize(True) or maximize(False)


# the simulator class
class Solver:
	def __init__(self, max_iter, problm):
		self.T = max_iter
		self.t = 0 			# initial iteration = 0
		self.problem = problm 	#Problem class obj

	#we pass FSS parameters here
	def initializeFSS(self, wscale, stepvol, stepind, stepins ):



	def solve(self):
"""

# HELPER FUNCTIONS


def generateRandList(dim, lower, upper, constraints=None):
    # we need to change the dtype of X from int to float here
    X = np.random.rand(dim)
    X = (X * (upper-lower)) + lower
    print(X)
    # and some generation magic here possibly by partially solving the system of Lin. inequalities
    return X


# incomplete should return euclidean distance between n1 and n2 vectors
def get_euclidean_dist(n1, n2):
    n3 = n1 - n2
    return np.linalg.norm(n3)


# TEST Functions == Objective functions to be optimized
# test function 1 = bell curve = exp(-((x+5)^2+y^2))
# global maxima is at (-5.0, 0)
def testFunction1(X):
    if isinstance(X, np.ndarray):
        return math.exp(-((X[0]+5)*(X[0]+5) + X[1]*X[1]))
    else:
        return None

# small test for Fish class
# its performing quite well even with only individual displace component and 1 fish
# Solutions produced are quite close to (-5.0, 0)


def main():
    # x1 = generateRandList(2, -10, 10)
    # w_scale = 40.0
    # step_indiv = 1.0
    # f1 = Fish(2, x1, w_scale/2, testFunction1)
    # f1.print_fish_status()
    T = 30
    # for i in range(0,T):
    #     f1.displace_ind(step_indiv)
    #     f1.update_fitness()
    #     f1.feed((f1.f - f1.f_prev), w_scale)
    s = School(5, 2, 40.0, testFunction1, 3.0, 2.0)
    s.init_fish_school()
    for i in range(0,T):
        s.update_school()
    s.print_school_info()
    return 0


if __name__ == "__main__":
    main()
