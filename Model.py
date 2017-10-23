####################################################################
## Import python modules
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as T
import theano
from theano.tensor.shared_randomstreams import RandomStreams
# theano.config.optimizer = "None"


####################################################################
## Parameters

dt 		= 	0.01		# time interval for ODE solver
N 		= 	300			# number of timesteps for ODE solver
S 		= 	3		 	# number of network species (genes)


####################################################################
# Model definition
class network:

	# this class and the functions in it can be modified to generalize the model

	def __init__(self,W=np.random.normal(0,0.1,[S,S])):
		# variables in model	
		self.W  = theano.shared(value=W.astype(theano.config.floatX), name='W', borrow=True)	
		self.Ax = theano.shared(1., name="Ax")	

		# list of parameters to optimize
		self.parameters = [self.W, self.Ax]													

	def networkFunction(self,y,x):
		# specification of network ODE, in the form: dy/dt = networkFunction(y,x)
		# y = vector of genes, x = vector of inputs
		tmp = T.dot(self.W,y)
		return 1/(T.exp(-tmp)+1)-y+x

	def normalize(self,y):
		# rescale outputs by Ax
		return self.Ax*y

	def regularize(self):
		# regularization function
		return T.mean(T.abs_(self.W))


def desiredFunction(B):
	# generates a new batch of B input/output data pairs (plus initial conditions)

	#initial condition, y0
	y0 = 0.1*np.ones([S,B]).astype(theano.config.floatX) * np.random.normal(loc = 1.0, scale = 0.001, size = [S,B])

	# network input, x, across all time points and genes
	x0 = np.random.uniform(0,2,B).astype(theano.config.floatX)
	x = x0.reshape(1,B) * np.random.normal(loc = 1.0, scale = 0.1, size = [N,S,B])      #we add noise to the input
	x[:,1:S,:] = 0.

	#desired output, y_
	y_ = np.zeros_like(x0)
	y_[x0>0.5] = 1
	y_[x0>1.5] = 0

	return [y0,x,y_]

def inputData(B):
	# generates a new batch of B input/output data pairs (plus initial conditions)

	#initial condition, y0
	y0 = 0.1*np.ones([S,B]).astype(theano.config.floatX) * np.random.normal(loc = 1.0, scale = 0.001, size = [S,B])

	# network input, x, across all time points and genes
	x0 = np.linspace(0,2,B).astype(theano.config.floatX)
	x = x0.reshape(1,B) * np.random.normal(loc = 1.0, scale = 0.1, size = [N,S,B])      #we add noise to the input
	x[:,1:S,:] = 0.

	#desired output, y_
	y_ = np.zeros_like(x0)
	y_[x0>0.5] = 1
	y_[x0>1.5] = 0

	return [y0,x,y_]

