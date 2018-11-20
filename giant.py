# giant_neural_networks's "Beginner Intro to Neural Networks" Youtube Series
from matplotlib import pyplot as plt
import numpy as np

# [length, width, type(0=blue, 1=red)]
data = [
	[3,   1.5, 1],
	[2,   1,   0],
	[4,   1.5, 1],
	[3,   1,   0],
	[3.5, 0.5, 1],
	[2,   0.5, 0],
	[5.5, 1,   1],
	[1,   1,   0]
]

# Based on the data given, the NN will try to
# predict what type the mystery flower is
mystery_flower = [4.5, 1]


# Neural Network
#   o	: flower type (0=blue, 1=red)
#  / \	: w1, w2, b
# o   o : length, width

w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def sigmoid_p(x):
	return sigmoid(x) * (1-sigmoid(x))

# T = np.linspace(-5, 5, 10)
# Y = sigmoid(T)

learning_rate = 0.5
costs = []

# Training loop
for i in range(50000):
	ri = np.random.randint(len(data)) # Random index
	point = data[ri]

	z = point[0] * w1 + point[1] * w2 + b
	pred = sigmoid(z) # Activation function. pred=prediction

	target = point[2]
	cost = np.square(pred - target)

	dcost_pred = 2 * (pred - target)
	dpred_dz = sigmoid_p(z)
	dz_dw1 = point[0]
	dz_dw2 = point[1]
	dz_db = 1
	dcost_dz = dcost_pred * dpred_dz

	dcost_dw1 = dcost_dz * dz_dw1
	dcost_dw2 = dcost_dz * dz_dw2
	dcost_db = dcost_dz * dz_db

	w1 = w1 - learning_rate * dcost_dw1
	w2 = w2 - learning_rate * dcost_dw2
	b = b - learning_rate * dcost_db


def predict(point):
	print(point)
	z = point[0] * w1 + point[1] * w2 + b
	pred = sigmoid(z)
	type = "red"
	if(pred < 0.5):
		type = "blue"
	print("prediction: {} \n".format(type))

# Predictions
for i in range(len(data)):
	point = data[i]
	predict(point)

predict(mystery_flower)
