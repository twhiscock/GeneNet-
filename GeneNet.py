
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as T
import theano
from theano.tensor.shared_randomstreams import RandomStreams
from Model import dt
# theano.config.optimizer = "None"


###################################################################
# Optimizer 

def AdamOptimizer(cost, params, lr=0.1, b1=0.02, b2=0.001, e=1e-8):
	updates = []
	grads = T.grad(cost, params)
	i = theano.shared(np.float32(1))
	i_t = i + 1.
	fix1 = 1. - (1. - b1)**i_t
	fix2 = 1. - (1. - b2)**i_t
	lr_t = lr * (T.sqrt(fix2) / fix1)
	for p, g in zip(params, grads):
		m = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX))
		v = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX))
		m_t = (b1 * g) + ((1. - b1) * m)
		v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
		g_t = m_t / (T.sqrt(v_t) + e)
		p_t = p - (lr_t * g_t)
		updates.append((m, m_t))
		updates.append((v, v_t))
		updates.append((p, p_t))
	updates.append((i, i_t))
	return updates


####################################################################
# Define dynamics of the network using theano.scan 

def evolveTime(initial, input_,network):
	result, updates = theano.scan(
			fn = lambda i, o: o + dt*network.networkFunction(o,i),
			sequences = input_,
			outputs_info = initial)
	return result[-1][-1][:]
	#returns final state

def fullEvolveTime(initial, input_,network):
	result, updates = theano.scan(
			fn = lambda i, o: o + dt*network.networkFunction(o,i),
			sequences = input_,
			outputs_info = initial)
	return result
	#returns entire time trace


####################################################################
# Define error function(s)

def costFunction(initial_, input_, output_,network):
	# unregularized cost 
	y = network.normalize(evolveTime(initial_, input_,network))
	y_ = output_
	return T.sum(((y-y_)**2)) 

def costFunctionL1(initial_, input_, output_,network,L1):
	# regularized cost 
	y = network.normalize(evolveTime(initial_, input_,network))
	y_ = output_
	L1norm = L1 * network.regularize()
	return T.sum(((y-y_)**2)) + L1norm 

####################################################################
# Execute training of network

def trainNetwork(desiredFunction, network, iterations, batchSize,mode = 'default',L1=0.1, pruneLimit = 1.0, plotCost = False):


	#build computational graph
	initial = T.matrix(name='initial',dtype=theano.config.floatX)
	networkInput = T.tensor3(name='networkInput',dtype=theano.config.floatX)
	desiredOutput = T.vector(name='desiredOutput',dtype=theano.config.floatX)
	L1mask = T.matrix(name='L1mask',dtype=theano.config.floatX)
	applyL1mask = theano.function(inputs = [L1mask],
					updates = [(network.W, network.W*L1mask)])
	simulationOutput = evolveTime(initial, networkInput,network)
	simulate = theano.function(
					inputs = [initial, networkInput],
					outputs = simulationOutput)
	fullSimulationOutput = fullEvolveTime(initial, networkInput,network)
	fullSimulate = theano.function(
					inputs = [initial, networkInput],
					outputs = fullSimulationOutput)

	cost = costFunction(initial, networkInput, desiredOutput,network)
	train = theano.function(
		inputs = [initial, networkInput, desiredOutput], 
		outputs = cost, 
		updates =  AdamOptimizer(cost, network.parameters))
	costL1 = costFunctionL1(initial, networkInput, desiredOutput,network,L1)
	trainL1 = theano.function(
		inputs = [initial, networkInput, desiredOutput], 
		outputs = costL1, 
		updates =  AdamOptimizer(costL1, network.parameters))



	# train network
	costValue = []
	iteration = 0
	if mode is 'default':
		while (iteration < iterations):
			[initial,networkInput,desiredOutput] = desiredFunction(batchSize)
			current_cost = train(initial, networkInput, desiredOutput)
			costValue = np.append(costValue, current_cost.item())
			iteration += 1
		iteration = 0
		while (iteration < iterations):
			[initial,networkInput,desiredOutput] = desiredFunction(batchSize)
			current_cost = trainL1(initial, networkInput, desiredOutput)
			costValue = np.append(costValue, current_cost.item())
			iteration += 1

		mask = np.abs(network.W.eval()) > pruneLimit
		iteration = 0
		while (iteration < iterations):
			[initial,networkInput,desiredOutput] = desiredFunction(batchSize)
			current_cost = train(initial, networkInput, desiredOutput)
			applyL1mask(mask)
			costValue = np.append(costValue, current_cost.item())
			iteration += 1




	if mode is 'vanilla':
		while (iteration < iterations):
			[initial,networkInput,desiredOutput] = desiredFunction(batchSize)
			current_cost = train(initial, networkInput, desiredOutput)
			costValue = np.append(costValue, current_cost.item())
			iteration += 1


	if mode is 'regularize':
		while (iteration < iterations):
			[initial,networkInput,desiredOutput] = desiredFunction(batchSize)
			current_cost = trainL1(initial, networkInput, desiredOutput)
			costValue = np.append(costValue, current_cost.item())
			iteration += 1


	if mode is 'prune':
		mask = np.abs(network.W.eval()) > pruneLimit
		while (iteration < iterations):
			[initial,networkInput,desiredOutput] = desiredFunction(batchSize)
			current_cost = train(initial, networkInput, desiredOutput)
			applyL1mask(mask)
			costValue = np.append(costValue, current_cost.item())
			iteration += 1


	if (plotCost):
		plt.plot(np.linspace(0,1,costValue.size),costValue)
		plt.show()

	return network.parameters


def simulateNetwork(inputData, network, batchSize):


	initial = T.matrix(name='initial',dtype=theano.config.floatX)
	networkInput = T.tensor3(name='networkInput',dtype=theano.config.floatX)
	desiredOutput = T.vector(name='desiredOutput',dtype=theano.config.floatX)
	simulationOutput = evolveTime(initial, networkInput,network)
	simulate = theano.function(
					inputs = [initial, networkInput],
					outputs = simulationOutput)
	fullSimulationOutput = fullEvolveTime(initial, networkInput,network)
	fullSimulate = theano.function(
					inputs = [initial, networkInput],
					outputs = fullSimulationOutput)

	[initial,networkInput,desiredOutput] = inputData(batchSize)
	outputSimulation = fullSimulate(initial, networkInput)

	return outputSimulation



