#binary FSS (domain is binary == sequence of 0s n 1s)
#problems being solved are discrete in nature (0/1 MKP belongs to this category)
#the fish class here has different operators(basically the displacement n feed works differently)

###INCOMPLETE

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
#from mpl_toolkits.mplot3d import Axes3D
from random import random,randint
import copy   
import time
import math
from Problem import Problem
import dat_parser as parser
from colour import Color


class School(object):
    def __init__(self, iter, problem, population_size, dim, objtve, step_ind, thresh_c, thresh_v):
        self.problem = problem
        self.max_iter = iter
        self.curr_iter = 0
        self.f_avg = 0.0
        self.size = population_size
        self.dim = dim
        self.w_scale = float(self.max_iter / self.dim)
        print('W scale = ',self.w_scale)
        self.school = []
        self.plot_data_xy = np.ndarray((self.max_iter * self.size,2),dtype=float)
        #self.stats = np.zeros(self.max_iter, dtype=float, order='C')
        self.prev_weight = self.w_scale/2 * self.size
        self.curr_weight = self.w_scale/2 * self.size
        self.step_ind = step_ind
        self.step_ind_init = step_ind
        self.thresh_c = thresh_c
        self.thresh_v = thresh_v
        self.del_f_max = 0.0				#school max fitness gain
        self.f_max = 0.0
        self.f_min = 0.0
        self.best_fish = None
        self.best_fish_global = None 			
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
        #self.update_stats(iter)
        while self.curr_iter < self.max_iter:
            self.update_plotdata()
            for i in self.school:
                i.displace_ind()
                #apply fitness function
                i.update_del_f()
                #i.print_fish_status()
            #update the best fish in school(according to current fitness)
            self.update_del_f_max()
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
            #self.update_del_f_max()
            self.update_best_fish()
            self.update_step_ind()
            self.curr_iter += 1
        
    def update_plotdata(self):
        #print(self.plot_data.shape)
        for i in range(0,self.size):
            nb = binaryseqtoi(self.school[i].X,self.dim)
            #theta = (2*math.pi/self.size) * i
            if nb == 0:
                radius = 0
            else:
                radius = math.log2(nb)
            self.plot_data_xy[self.curr_iter * self.size + i, 0] = radius
            self.plot_data_xy[self.curr_iter * self.size + i, 1] = self.school[i].f

    def update_del_f_max(self):
        max = 0
        for i in range(0,self.size):
            if self.school[i].del_f > self.school[max].del_f:
                max = i
        self.del_f_max = self.school[max].del_f
    
    def update_step_ind(self):
        self.step_ind = self.step_ind - self.step_ind_init / self.max_iter
    
    def update_f_avg(self):
        self.f_avg = 0.0
        for i in range(0,self.size):
            self.f_avg += self.school[i].f
        self.f_avg /= self.size
    # def update_stats(self, iter):
    #     self.update_f_avg()
    #     self.stats[iter] = self.f_avg
    #     #plt.plot(self.stats)
    #     #plt.show()



    def update_col_ins_vec(self):
        sigma_del_f = 0.0
        for i in self.school:
            sigma_del_f += i.del_f
        self.col_ins_disp.fill(0.0)
        for i in self.school:
            self.col_ins_disp += i.del_f * (i.X)    #important change instead of del x we use x itself
        if sigma_del_f:
            self.col_ins_disp *= 1/sigma_del_f
        else:
            self.col_ins_disp.fill(0.0)
        


    def update_best_fish(self):
        max_curr = 0
        max_global = -1
        for i in range(0,self.size):
            if self.school[i].f > self.f_max:
                self.f_max = self.school[i].f
                max_global = i
            if self.school[i].f > self.school[max_curr].f:
                max_curr = i
        self.best_fish = self.school[max_curr] 
        if(max_global >= 0):
            self.best_fish_global = np.copy(self.school[max_global].X)
        #print('Best Solution = ', self.best_fish_global, ' Best Fitness = ', self.f_max)
        #print('Best Solution Current = ', self.best_fish.X, ' Current Best Fitness = ', self.best_fish.f)


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
        #print(self.barycenter)
        


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
            np.copyto(self.X, m, casting='same_kind', where=True)

    def update_del_f(self):
        self.del_f = self.f - self.f_prev


    def feed(self):
        # weight can decrease if fitness decreases
        if(self.school.del_f_max):
           self.W += (self.del_f)/abs(self.school.del_f_max)
        self.W = min(self.W, self.school.w_scale)
        #print(self.W)

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
        np.copyto(self.X, temp, casting='same_kind', where=True)
        self.f = getObjective(self.X, self.school.objective)
        
        
        
    def displace_col_vol(self):
        d = randint(0,self.school.dim - 1)
        temp = np.copy(self.X)
        if temp[d] != self.school.barycenter[d]: #random bit that is not same is changed
            if(self.school.curr_weight - self.school.prev_weight > 0):
                temp[d] = self.school.barycenter[d]
            else:
                temp[d] = not self.school.barycenter[d]
        np.copyto(self.X_prev, self.X, casting='same_kind', where=True)
        self.f_prev = self.f        
        if check_constraints_linear(temp, self.school.problem.constraints, self.school.problem.bounds) == False:
            return
        np.copyto(self.X, temp, casting='same_kind', where=True)
        self.f = getObjective(self.X, self.school.objective)
            

    # # debug functions
    # def print_fish_status(self):
    #     print("X = ", self.X)
    #     print("X prev = ", self.X_prev)
    #     print("Weight = ", self.W)
    #     print("fitness = ", self.f)
    #     print("Prev fitness = ", self.f_prev)
    #     print("y = ", self.y)
    #     print("y prev = ", self.y_prev)

        
class Solver:
    def __init__(self, runs, iterations, problem, population_size, step_ind, thresh_c, thresh_v):
        self.runs = runs
        self.T = iterations
        self.t = 0  #current interation
        self.problem = problem
        self.population = population_size
        #self.w_scale = w_scale
        self.step_ind = step_ind
        self.thresh_c = thresh_c
        self.thresh_v = thresh_v
        self.school = School(self.T, self.problem, self.population, self.problem.dim, self.problem.objective, self.step_ind, self.thresh_c, self.thresh_v)


class Stats:
    pass
# HELPER FUNCTIONS

#this function affects solution quality

def generateRandBinSeq(dim, constraints=None, bounds=None):
    # we need to change the dtype of X from int to uint
    X = np.zeros(dim, dtype=np.uint8, order='C')
    #print(X.size)
    for i in range(0, X.size):
        if(random() >= 0.5):
            X[i] = 1
    d = randint(0,dim-1)
    while(check_constraints_linear(X, constraints, bounds) == False):
        #print('X before mutation ', X)
        if(X[d] == 1):
            X[d] = 0
        d += 1
        if(d >= dim):
            d = 0
        #print('X after mutation ', X)
    #print(X)
    return X

def getObjective(X, objective):
    return X.dot(objective)         #assuming it will return their inner product

def binaryseqtoi(bin_arr, len):
    num = 0
    #print(type(bin_arr[0].item())) # numpy dtypes suck
    for i in range(len-1,-1,-1):
        num = num * 2 + bin_arr[i].item()
        #print(int(bin_arr[i]))
    #print(num)
    return num


def get_euclidean_dist(n1, n2):
    n3 = n1 - n2
    return np.linalg.norm(n3)


def check_constraints_linear(X, coef, bounds):
    for i in range(0, len(coef)):
        linear_sum = X.dot(coef[i])
        if linear_sum > bounds[i]:
            return False
    return True

###For animation


def animate(i,data):
    scat.set_offsets(data[i*pop:(i+1)*pop,:])
    return scat,

f_path = '../test/datasets/MKP/chubeas/OR30x100/OR30x100-0.25_10.dat'
p = parser.parse_single_instance(f_path)
print(p.bounds)
T = 10000
pop = 30
s = Solver(1, T, p, pop, 0.5, 0.4 ,0.4)
init_time = time.time()
s.school.init_fish_school()
init_time = time.time() - init_time
print('Initialization completed')
print('Time taken to initialize = ', init_time, ' seconds')
print('Population = ', pop)
print('Dimensions = ', p.dim)
print('No. of constraints = ', p.n_constraint)
print('Max Iterations = ', T)
print('Running Simulation...')
simu_time = time.time()
s.school.update_school()
simu_time = time.time() - simu_time
print('Algorithm run time = ', simu_time, ' seconds')
print('Best Solution = ', s.school.best_fish_global)
print('Best Fitness = ', s.school.f_max)
print('Average Fitness = ', s.school.f_avg)
print(check_constraints_linear(s.school.best_fish_global, s.problem.constraints, s.problem.bounds))
fig = plt.figure()
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
ax = plt.axes(xlim=(0, s.school.dim*2), ylim=(np.amin(s.school.plot_data_xy[:, 0]),np.amax(s.school.plot_data_xy[:, 1])*2))
scat = ax.scatter(x=s.school.plot_data_xy[0:pop,0],y=s.school.plot_data_xy[0:pop,1],marker='.')
anim = FuncAnimation(fig, animate, fargs=(s.school.plot_data_xy,),frames=T, blit=True,cache_frame_data=False)
anim.save('plot_animation.mp4', writer=writer)
print('Saved animation!')

