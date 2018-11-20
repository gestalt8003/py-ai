# Feed-Forward Neural Network
#   o	: flower type (0=blue, 1=red)
#  / \	: w1, w2, b
# o   o : length, width

import numpy as np
from nn_01_utils import *


LAYERS = [1] # 1 layer(out) w/ 1 nodes
NUM_INPUTS = 2
LEARNING_RATE = 0.5
TRAINING_LOOPS = 50000


# The data that we are using to train the network
# [length, width]
data = np.array([
	[3,   1.5],
	[2,   1  ],
	[4,   1.5],
	[3,   1, ],
	[3.5, 0.5],
	[2,   0.5],
	[5.5, 1  ],
	[1,   1  ]
])

# What we want the network to output
# [type(0=blue, 1=red)]
targets = np.array([
	[1, 0, 1, 0, 1, 0, 1, 0]
])

print('DATA:\n', data)

# Layers
# layers[l][n][v] (l=layer #, n=node v=value(0=weights, 1=bias))
# layers[l][n][0][w] (w=weight for node w)
layers = generate_layers(LAYERS, NUM_INPUTS)

def calculate(point):
	result = point
	# Send data through neural network
	for l in range(len(layers)):
		layer = layers[l]
		# Get result from previous layer (if 1st, use inputs)
		next_result = list(range(len(layer)))
		for i in range(len(layer)):
			node = layer[i]
			z = 0 # What will eventually be the result for this node
			# Apply weights to values
			weights = node[0]
			for w in range(len(weights)):
				z += result[w] * weights[w]
			bias = node[1]
			z += bias
			# Apply acivation function
			next_result[i] = sigmoid(z)
		result = next_result
	return result[0]

def train(times):
	# Training loop
	for i in range(times):
		# Get random point from data to train off of
		ri = np.random.randint(len(data))
		point = data[ri]

		result = calculate(point)

		# Correct errors
		target = targets[0][ri] # TODO make modular
		error = target - result

		dsum = sigmoid_p(result) * error # Delta output sum
		for layer in layers:
			for node in layer:
				weights = node[0]
				dweights = [dsum / w for w in weights]
				for i in range(len(weights)):
					weights[i] -= dweights[i]

train(TRAINING_LOOPS)

def predict(point):
	print(point)
	pred = calculate(point)
	type = "red"
	if(pred < 0.5):
		type = "blue"
	print("prediction: {} \n".format(type))

# Predictions
for i in range(len(data)):
	point = data[i]
	predict(point)
