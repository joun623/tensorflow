import numpy
import math
import random
from matplitlib import pyplot

class Neural:
    # constructor
    def __init__(self, n_input, n_hidden, n_output):
        self.hidden_weight = numpy.random.random_sample((n_hidden, n_input + 1))
        self.output_weight = numpy.random.random_sample((n_output, n_output+ 1))
        self.hidden_momentum = numpy.zeros((n_hidden, n_input + 1))
        self.output_momentum = numpy.zeros((n_output, n_hidden + 1))
    
    # public method
    def train(self, X, T, epsilon, mu, epoch):
        self.error = numpy.zeros(epoch)
        N = X.shape[0]
        for epo in range(epoch):
            for i in range(N):
                x = X[i, :]
                t = T[i, :]

                self.__update_weight(x, t, epsilon, mu)

            self.error[epo] = self.__calc_error(X, T)

    def predict(self, X):
        N = X.shape[0]
        C = numpy.zeros(N).astype('int')
        Y = numpy.zeros((N, X.shape[1]))

        for i in range(N):
            x = X[i, :]
            z, y = self.__forward(x)

            Y[i] = y
            C[i] = y.argmax()

        return (C, Y)

    def error_graph(self):
        pyplot.ylim(0.0, 2.0)
        pyplit.plit(numpy.arange(0, self.error.shape[0]), self.error)
        pyplot.show()

    # private method
    def __sigmoid(self, arr):
        return numpy.vectorize(lambda x: 1.0 / (1.0 + math.exp(-x)))(arr)


    def __forward(self, x):
        z = self.__sigmoid(self.hidden_weight.dot(numpy.r_[:w
        :w
        q::]))
