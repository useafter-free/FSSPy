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


# class Fish(object):
# 	def __init__(self, dim, w, school_ref):
#     	self.dim = dim
#     	self.X = [0 for _ in range(self.dim)]
#     	self.X_prev =  [0 for _ in range(self.dim)]	#dont know if its necessary
#     	self.Y_curr = 0
#     	self.Y_prev = 0
#         #self.id = a
#         self.W = w 		# initial weight of each fish is = wscale/2
#         				# which is a school parameter
#         self.f = 0 		#not sure if I should initialize it t 0
#         self.school = school_ref

#     def updateFish(self):


#     def set_pos(self, pos):
#     	self.X = X

#     def swim(self):
#             #Attempt at indvidual movement here
#          for fish in self.school:
#             new_pos = np.zeros((self.dim,), dtype=np.float)
#             for dim in range(self.dim):
#                 newpos[dim] = fish.pos[dim] + (self.newmovement * np.random.uniform(-1, 1))
#             #fitness is a predefined criteria here.
#             if fitness < fish.fitness:
#                 fish.change_in_fitness = abs(newfitness - fitness)
#                 fish.fitness = fitness

#                 for idx in range(self.dim):

#                 fish.pos = newpos
#             else:
#                pass
#     	#indiviual component of movement/soln search

#     def eat():
#     	#update weight basically/evaluate last swim == soln feasibility
#     	#weight calc depends on curr fitness which depends on prev n curr Y measure of the fish

#     def follow_the_school(self):
#     	#Fitness-based component
# == collective-instinctive component of movement/soln search
# basically finds weighted avg displacement of the school
# net fitness/improvement is the deciding factor
# fitness is directly dependent on Y measure of X of each fish
# it doesnt involve curr or prev W of the fish or school

# Weight-based component
# == collective-volitive component of movement/soln search
# basically spread or shrink based on based school health/weight improvement
# shrink if school weight improved(basically move towards barycenter)
# else spread : exploration component of the search (has random component)


# class School(object):
#     def __init__(self, schoolsize, dim, wscale):
#     	self.size = schoolsize
#     	self.wscale = wscale
#     	self.school = [Fish(self.dim, self.wscale/2) for _ in range(self.size)]
#     	self.prev_weight = 0.0
#     	self.curr_weight = 0.0
#     	self.f_school = 0				#school fitness
#     	self.best_fish = None 			#fish with max fitness
#     	self.y_measure = None 			#will be a function ptr or lambda


#     def init_fish_school(self, pos):
#         fish = Fish(self.dim)
#         fish.pos = pos
#         fish.weight =


#     def updateFitness(self):
#     	a = self.curr_weight
#     	if a<15: 					#why 15? we need to define n initialize constants n param
#     		self.f_school -= 1
#     	else:
#     		self.f_school += 1

#     		#idk what this is
#     def checkfitnessswarm(self, f, n):
#     	z= 1
#     	if fswarm/n<w*n/2:
#     		z=0

#     #school of fish should be a bin maxHeap with fish W as the key
#     def updateBestFish(x):

#     #basically heapify the new fish school list
#     	pass

#     def updateBarycenter(self):
#     	self.barycenter = [0 for _ in range(self.dim)]
#     	for i in range(self.size):
#     		for j in range(self.dim):
#     			self.barycenter[j] += (self.school[i][j]*self.school[i].W)/self.curr_weight

#import check_constraints as strain
class Fish:
    def __init__(self, dim, x, w, fun):
        self.dim = dim
        self.X = x
        self.X_prev = np.copy(self.X)
        self.W = w
        self.f = 1.0
        self.f_prev = 1.0
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
        self.f += self.y - self.y_prev

    def feed(self, del_f_max, w_scale):
        # weight can decrease if fitness decreases
        if(del_f_max):
           self.W += (self.f - self.f_prev)/abs(del_f_max)
        self.W = min(self.W, w_scale)

    def displace_col_ins(self, m):
        #m = np.zeroes(self.dim, dtype=float, order='C')
        # is it just holding a reference (beocz then i will have to copy it everytime)
        np.copyto(self.X_prev, self.X, casting='same_kind', where=True)
        self.X += m  # numpy arr can be added like matrices ryt?

    def displace_col_vol(self, bary, step_vol):
        distance = get_euclidean_dist(self.X, bary)
        np.copyto(self.X_prev, self.X, casting='same_kind', where=True)
        # again idk if we can scale numpy array like matrices
        self.X += (step_vol/distance) * (self.X - bary)

    # debug functions
    def print_fish_status(self):
        print("X = ", self.X)
        print("X prev = ", self.X_prev)
        print("Weight = ", self.W)
        print("fitness = ", self.f)
        print("Prev fitness = ", self.f_prev)
        print("y = ", self.y)
        print("y prev = ", self.y_prev)


# incomplete
class School:
    pass


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
    print(type(X))
    if isinstance(X, np.ndarray):
        return math.exp(-((X[0]+5)**2 + X[1]*X[1]))
    else:
        return None

# small test for Fish class
# its performing quite well even with only individual displace component and 1 fish
# Solutions produced are quite close to (-5.0, 0)


def main():
	x1 = generateRandList(2, -10, 10)
	w_scale = 40.0
	step_indiv = 1.0
	f1 = Fish(2, x1, w_scale/2, testFunction1)
	f1.print_fish_status()
	T = 5000
	for i in range(0,T):
		print("Iteration = ", i)
		f1.displace_ind(step_indiv)
		f1.update_fitness()
		f1.feed((f1.f - f1.f_prev), w_scale)
		f1.print_fish_status()
	return 0


if __name__ == "__main__":
    main()
