import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

####################################################################
##Parameters
B 		= 128    	#batch size
N 		= 300		#timesteps
S 		= 3		 	#number of species
dt 		= 0.01		#time interval
L1 		= 0.02 		#L1 regularization parameter

####################################################################
##Data 
input_ 		= tf.placeholder(tf.float32, shape=[N, B, S])
output_ 	= tf.placeholder(tf.float32, shape=[B])
initial_ 	= tf.placeholder(tf.float32, shape=[B, S])

####################################################################
##Variables to optimize
W = tf.Variable(tf.random_normal([S,S], mean = 0, stddev = 0.1, dtype = tf.float32))
A = tf.Variable(1.0, dtype = tf.float32)

####################################################################
##Nonlinearity chosen
def phi(x):
	return 1/(tf.exp(-x)+1)

####################################################################
##ODE function
def simulate(input_, initial_, W):
	output = tf.scan(lambda o,i: o + dt*(phi(tf.matmul(o,W))-o+i),
							elems = input_,
							initializer = initial_,
							swap_memory = True)
	return output


output = simulate(input_, initial_, W)

####################################################################
## Training function
relevantOutput 	= A*output[N-1,:,1]
## Without regularization
cost = tf.reduce_mean(tf.square(((relevantOutput - output_)))) 
train_step = tf.train.AdamOptimizer(learning_rate=0.2, beta1=0.98, beta2=0.999, epsilon=1e-08).minimize(cost)
## With regularization
costL1 = tf.reduce_mean(tf.square(((relevantOutput - output_)))) + L1*tf.reduce_mean(tf.abs(W)) 
train_stepL1 = tf.train.AdamOptimizer(learning_rate=0.2, beta1=0.98, beta2=0.999, epsilon=1e-08).minimize(costL1)

####################################################################
## Training data - here for a "stripe"
def newBatch(plot=False):
	initialVal		= 0.1*np.ones([B,S])
	if(plot):
		inputNoNoise = np.linspace(0,2,B)
	else:
		inputNoNoise    = np.random.uniform(0,2,B)
	inputVal		= inputNoNoise.reshape(B,1) * np.random.normal(loc=1.0, scale = 0.0001, size=[N,B,S])
	inputVal[:,:,1:S] = 0.0
	outputNoNoise  	= np.zeros_like(inputNoNoise)
	outputNoNoise[np.logical_and(inputNoNoise > 0.5,inputNoNoise < 1.5 )] = 1.0
	return [inputVal, outputNoNoise, initialVal]



####################################################################
## Training model function
def trainModel(iterations,regularize=False,prune=False,pruneLimit=1,print=False):
	mask = np.abs(sess.run(W)) > pruneLimit
	for i in range(iterations):
		[inputVal, outputVal, initialVal] = newBatch()
		if(regularize):
			sess.run(train_stepL1, feed_dict = {input_: inputVal, initial_: initialVal, output_: outputVal})
		else:
			sess.run(train_step, feed_dict = {input_: inputVal, initial_: initialVal, output_: outputVal})
		if(print):
			ww = sess.run(cost, feed_dict = {input_: inputVal, initial_: initialVal, output_: outputVal})
			print([ww, i])
		if(prune):
			applyMask = W.assign(W*mask)
			sess.run(applyMask)

####################################################################
## Simulate model
def simulateModel():
	[inputVal, outputVal, initialVal] = newBatch(plot=True)
	finalOutput = sess.run(relevantOutput, feed_dict = {input_: inputVal, initial_: initialVal, output_: outputVal})
	weights = sess.run(W)
	print(weights)
	plt.plot(finalOutput)
	plt.show()


####################################################################
## Train model

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
trainModel(iterations=2000)
simulateModel()
trainModel(iterations=1000, regularize=True)
simulateModel()
trainModel(iterations=1000, prune=True)
simulateModel()











