#!/usr/bin/env python

#import numpy as np
#import copy
import random
import check_constraints as strain



class Fish(object):
	def __init__(self, dim, w, school_ref):
    	self.dim = dim
    	self.X = [0 for _ in range(self.dim)]
    	self.X_prev =  [0 for _ in range(self.dim)]	#dont know if its necessary
    	self.Y_curr = 0
    	self.Y_prev = 0
        #self.id = a
        self.W = w 		# initial weight of each fish is = wscale/2
        				# which is a school parameter
        self.f = 0 		#not sure if I should initialize it t 0
        self.school = school_ref

    def updateFish(self):



    def set_pos(self, pos):
    	self.X = X

    def swim():
    	#indiviual component of movement/soln search

    def eat():
    	#update weight basically/evaluate last swim == soln feasibility
    	#weight calc depends on curr fitness which depends on prev n curr Y measure of the fish

    def follow_the_school(self):
    	#Fitness-based component
    	# == collective-instinctive component of movement/soln search
    	#basically finds weighted avg displacement of the school
    	#net fitness/improvement is the deciding factor
    	#fitness is directly dependent on Y measure of X of each fish
    	#it doesnt involve curr or prev W of the fish or school

    	
    	#Weight-based component
    	# == collective-volitive component of movement/soln search
    	#basically spread or shrink based on based school health/weight improvement
    	#shrink if school weight improved(basically move towards barycenter)
    	#else spread : exploration component of the search (has random component)




class School(object):
    def __init__(self, schoolsize, dim, wscale):
    	self.size = schoolsize
    	self.wscale = wscale
    	self.school = [Fish(self.dim, self.wscale/2) for _ in range(self.size)]
    	self.prev_weight = 0.0
    	self.curr_weight = 0.0
    	self.f_school = 0				#school fitness
    	self.best_fish = None 			#fish with max fitness
    	self.y_measure = None 			#will be a function ptr or lambda
     
    
    def init_fish_school(self, pos):
        fish = Fish(self.dim)
        fish.pos = pos
        fish.weight = 


    def updateFitness(self):
    	a = self.curr_weight
    	if a<15: 					#why 15? we need to define n initialize constants n param
    		self.f_school -= 1
    	else:
    		self.f_school += 1

    		#idk what this is 
    def checkfitnessswarm(self, f, n):
    	z= 1
    	if fswarm/n<w*n/2:
    		z=0

    #school of fish should be a bin maxHeap with fish W as the key      
    def updateBestFish(x): 

    #basically heapify the new fish school list
    	pass

    def updateBarycenter(self):
    	self.barycenter = [0 for _ in range(self.dim)] 
    	for i in range(self.size):
    		for j in range(self.dim):
    			self.barycenter[j] += (self.school[i][j]*self.school[i].W)/self.curr_weight

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
		self.opt_t = opt 					# minimize or maximize


class Solver:
	def __init__(self, max_iter, problm):
		self.T = max_iter
		self.t = 0 			# initial iteration = 0
		self.problem = problm 	#Problem class obj

	def initializeFSS(self):


	def solve(self):






def generateRandList(dim, constraints):
	X = random.sample(range(-3, 3), dim)
	#and some generation magic here possibly by partially solving the system of Lin. inequalities
	return X

def main():
	return 0

if __name__ == "__main__": main()
