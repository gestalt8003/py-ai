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

T = np.linspace(-5, 5, 10)
Y = sigmoid(T)
plt.plot(T, Y)

# Training loop
for i in range(1000):
	ri = np.random.randint(len(data)) # Random index 
	point = data[ri]
	print(point)
