"""
    Feed forward neural network library for MNIST handwritten digit
    recognition. The library has been created based on the book at
    http://neuralnetworksanddeeplearning.com by Michael Nielsen.

    Author: Zilvinas Verseckas 2017
"""

import numpy as np
import random

class Network(object):
    """
        Initialiser for a neural network. Creates a network with as many
        neurons as provided in the `sizes` vector and assigns random weights
        with biases to each neuron.
    """
    def __init__(self, sizes):
        self.sizes = sizes
        self.layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    """
        Passes the a `a` through the network.
    """
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    """
        Trains the network using a stochastic gradient descent method.
        The `training_data` is a list of tuples `(x, y)` representing
        an input vector and a desired target vector. If the `test_data`
        (which has the same structure as the `training_data`) is supplied
        the network's performance if evaluated at the end of each epoch.
    """
    def train(self, training_data, epochs, batch_size, eta, test_data = None):
        for j in range(epochs):
            # Randomize the training data and split it to batches
            random.shuffle(training_data)
            batches = [training_data[k:k + batch_size]
                       for k in range(0, len(training_data), batch_size)]
            for batch in batches:
                self.update_batch(batch, eta)

            # Performance evaluation and logging
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(
                    j, self.evaluate(test_data), len(test_data)))
            else:
                print('Epoch {0} complete'.format(j)) 
    """
        Tests a network precision
    """
    def test(self, test_data):
        print('Test result: {} / {}'.format(
            self.evaluate(test_data), len(test_data)))

    """
        Updates weights and biases for the network by applying gradient
        descent with back-propagation to a single batch.
    """
    def update_batch(self, batch, eta):
        nabla_b, nabla_w = self.zero_replicas()
        # Update the zero filled gradients for each tuple in a batch
        for a, y in batch:
            dt_nabla_b, dt_nabla_w = self.backprop(a, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, dt_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, dt_nabla_w)]

        self.weights = [w - (eta / len(batch)) * nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    """
        Performs the back-propagation on the network. Returns a tuple
        `(nabla_b, nabla_w)` representing the gradient for the cost function.
        Both `nabla_b` and `nambla_a` are layer-by-layer lists of arrays.
    """
    def backprop(self, a, y):
        nabla_b, nabla_w = self.zero_replicas()
        acts, zs = [a], []

        # Feed forward
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            zs.append(z)
            a = sigmoid(z)
            acts.append(a)

        # Backward pass
        dt = nabla_cost(acts[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = dt
        nabla_w[-1] = np.dot(dt, acts[-2].T)

        for l in range(2, self.layers):
            z = zs[-l]
            dt = np.dot(self.weights[-l + 1].T, dt) * sigmoid_prime(z)
            nabla_b[-l] = dt
            nabla_w[-l] = np.dot(dt, acts[-l - 1].T)

        return (nabla_b, nabla_w)

    """
        Evaluates how many correct guesses the network makes
    """
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    """
        Returns a tuple with zero filled replicas of of biasses and weights 
    """
    def zero_replicas(self):
        return ([np.zeros(b.shape) for b in self.biases],
                [np.zeros(w.shape) for w in self.weights])

"""
    Sigmoid function
"""
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

"""
    First derivative of sigmoid
"""
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

"""
    Gradient of a quadratic cost function.
    Returns a vector of partial derivatives d C/d a for the output
"""
def nabla_cost(a, y):
    return (a - y)
