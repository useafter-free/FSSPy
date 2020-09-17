import numpy as np
import copy
import random

# dim =(20,20) it is basically an array type thing.


class Fish(object):
    def __init__(self, dim):
        self.pos = [nan for _ in range(dim)]
        self.num = a
        self.weight=w


class fschool(object)
    def __init__(self, schoolsize, wscale)
         self.school_size = schoolsize
         self.wscale = wscale
         self.prev_weight_school = 0.0
         self.curr_weight_school = 0.0
         self.best_fish = None

         
    def __init_fish(self, pos):
        fish = Fish(self.dim)
        fish.pos = pos
        fish.weight = self.__gen_weight()




    def __weight():
      global w
      w = random.randint(20,30)

    def __checkfitness():
      a = w/2
      global fswarm
      if a<15:
          fswarm=fswarm-1
      else:
          fswarm=fswarm+1


    def __checkfitnessswarm(f, n):
      global z
      z= 1
      if fswarm/n<w*n/2:
          z=0

    def __bestfish(x):
     global bfish
     bfish = x

    def __calculate_barycenter(self):
        barycenter = np.zeros((self.dim,), dtype=np.float)
         density = 0.0

            for fish in self.fish:
                density += fish.weight
                for dim in range(self.dim):
                    barycenter[dim] += (fish.pos[dim] * fish.weight)
            for dim in range(self.dim):
                barycenter[dim] = barycenter[dim] / density

            return barycenter

    
def mainfunc():
    
    objs = list()
    for i in range(10):
        objs.append(Fish(i))


    while z !=0:


