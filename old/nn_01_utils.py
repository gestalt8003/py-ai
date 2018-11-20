# Useful functions for neural networks
import numpy as np

def generate_node(num_nodes_prev):
	# Initialize node array
	node = [[], []]
	# Generate weights
	for i in range(num_nodes_prev):
		node[0].append(np.random.randn())
	# Generate bias
	node[1] = np.random.randn()
	return node

def generate_layer(num_nodes, num_nodes_prev):
	# Initialize layer array
	layer = []
	# Generate nodes
	for i in range(num_nodes):
		layer.append(generate_node(num_nodes_prev))
	return layer

def generate_layers(d_layers, num_inputs):
	# Initialize layers array
	layers = []
	# Generate layers
	for i in range(len(d_layers)):
		if i == 0:
			num_nodes_prev = num_inputs
		else:
			num_nodes_prev = d_layers[i-1]
		layers.append(generate_layer(d_layers[i], num_nodes_prev))
	# Return as numpy array
	return np.array(layers)

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def sigmoid_p(x):
	return sigmoid(x) * (1-sigmoid(x))
