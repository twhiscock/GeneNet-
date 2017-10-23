####################################################################
#Example script to show the learning of a french flag circuit

####################################################################
#Import modules
## Import python libraries
import numpy as np     
import matplotlib.pyplot as plt
import theano.tensor as T
import theano

# Import GeneNet.py
import GeneNet

# Import Model.py, which specifies the network to be trained
import Model


####################################################################
# Run script


#Initialize network
network = Model.network()

#Train network
GeneNet.trainNetwork(Model.desiredFunction, network, 1000, 64, plotCost = True)

#Run simulation of network
output = GeneNet.simulateNetwork(Model.inputData, network, 10)

#Use simulation to plot output over time e.g.
y = output[-1,-1,:]
x = np.linspace(0,2,y.size)
plt.plot(x,y)
plt.show()










