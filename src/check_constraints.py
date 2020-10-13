#!/usr/bin/env python

"""
2 types of contraints:
->Hard constraints(MUST be satisfied)(this one I know)
Sub-categories in Hard constraints:
-->Equality constraint
-->Inequality constraint
(even in inequality constraints we will be do only the
Linear inqualities...which we can extend to Polynomial inequalities)
->Soft constraints(not covering in this module:( )
"""

"""
So how do we represent a Linear inequality....using a list maybe

for a input tuple X ===(x1,x2,....xn)
we have the Linear inequalities of the form
a1x1+a2x2+....+anxn <= 0
corresponding to this we can have a list
I=[a1,a2,....,an]
representing the Linear inequality
then we simply check 
sum = 0
for i in 1 to n: 
	sum+=I[i]*X[i]
if sum <= 0:
	return True
else :
	return False
BINGO!

same we do for Equality constraints just sum it over and check if its 0
we can read the inquality lists from .dat files
parse them then check if certain X satisfies it or not

"""

def check_constraints_linear(X, constraints):
	if constraints == None or X == None:
		return None 
	for i in range(0,constraints.length()):
		linear_sum = 0
		if(isinstance(constraints[i], list) == False or constraints[i].length != X.length):
			print("Malformed Input")
			return None
		for j in range(0,X.length()):
			linear_sum += linear_sum + constraints[i][j] * X[j]
		if linear_sum > 0:
			return False
	return True

def main():
	# dat_file as input  --> dat_parser --> X and constraints
	# but for since we dont have the parser we will just use manual input 
	# or even lazier hard coded X and constraints 


if __name__ == "__main__":
	main()