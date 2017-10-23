To use GeneNet, you must:						(example:)
(1) specify a model 							(Model.py)
(2) specify a desired function 					(Model.py)
(3) import GeneNet and theano 					(GeneNet.py)
(4) train / simulate the model using GeneNet 	(exampleScript.py)


There are two key functions that GeneNet executes: (1) train the network to perform some desired function using "trainNetwork"; and (2) simulate the network and examine its behaviour, using "simulateNetwork". These are explained below.


############################################################################################

GeneNet.trainNetwork()

FUNCTION: fit the parameters of a network model to perform some desired function

INPUT ARGUMENTS:
 - desiredFunction:
 		a python function that generates training data for the network, specifically:
 			1. initial conditions for network, 	y0
 			2. inputs to the network, 			x
 			3. desired network output, 			y_
 		returns [y0,x,y_]
 - network:
 		a python class that specifies the model to be trained, including:
 			1. initialization of the model 				(def __init__)
 			2. a list of parameters to be optimized		(network.parameters)
 			3. the equations governing network dynamics	(network.networkFunction)
 			4. a regularization function  				(network.regularize)
 			5. a normalization function 				(network.normalize)
 - iterations:
 		the number of steps to perform in the optimization procedure
 

OPTIONAL ARGUMENTS

 - mode:
 		if none, or 'default' is specified, then optimization is in three phases: (1) without regularization, (2) with regularization, and (3) with a pruned network

 		other modes are:
 			1. 'vanilla'  	(unregularized)
 			2. 'regularize' (regularized)
 			3. 'prune'		(pruned network)

 - L1:
 		regularization parameter (lambda in paper)
 		default = 0.1

 - pruneLimit:
 		pruning parameter (epsilon in paper) 
 		If mode is 'prune' (or in stage (3) of 'default'), then network parameters below 		this value are set to zero 
 		default = 1.0

 - plotCost:
 		if 'true', plot the evolution of cost vs. iteration number after optimization
 		default: 'false'

 OUTPUTS

 None. However, the parameters in the network are updated. 



############################################################################################

GeneNet.simulateNetwork()

FUNCTION: simulate a network model

INPUT ARGUMENTS:
- inputData
 		a python function that generates the necessary simulation inputs for the network,
 		specifically:
 			1. initial conditions for network, 	y0
 			2. inputs to the network, 			x
 			3. desired network output, 			y_ (optional)
 		returns [y0,x,y_] 
 		note that this can be identical, or different, to the desiredFunction input in GeneNet.trainNetwork()
 		
 - network:
 		the same model as specified in the GeneNet.trainNetwork() inputs

 - batchSize:
 		the number of inputs to simulate over





 		