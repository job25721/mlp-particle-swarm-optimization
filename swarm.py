from functions import printProgressBar
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.e**(-x))


class Node:
    def __init__(self):
        self.w = []
        self.b = 0.0
        self.input = []
        self.y = 0

    def initialWeight(self, n):
        self.w = np.random.uniform(-10, 10, size=n).reshape(n, 1)
        # self.b = np.random.uniform(-1, 1)

    def addInput(self, inputVector):
        self.input = inputVector

    def calculate_output(self, net):
        if len(net) != 0 and len(self.w) != 0:
            self.y = sigmoid(np.array(net).dot(self.w)[0] + self.b)
        else:
            self.y = np.array(self.input)


class Model:
    def __init__(self):
        self.particle = []
        self.x = []
        self.pbest = float('inf')
        self.xpbest = []
        self.v = []

    def feed_forward(self, inputVector):
        for l, layer in enumerate(self.particle):
            for i, node in enumerate(layer):
                if l == 0:
                    node.addInput(inputVector[i])
                    node.calculate_output([])
                else:
                    net = [prev.y for prev in self.particle[l-1]]
                    node.calculate_output(net)

    def create(self, inputLayer, hiddenLayers, outputLayer):
        self.inputLayer = inputLayer
        self.hiddenLayers = hiddenLayers
        self.outputLayer = outputLayer

        self.particle.append(self.inputLayer)

        for layer in self.hiddenLayers:
            self.particle.append(layer)
        self.particle.append(self.outputLayer)

        for node in self.outputLayer:
            node.initialWeight(
                n=len(self.hiddenLayers[len(self.hiddenLayers) - 1]))
            for w in node.w:
                self.x.append(w)

        for l, layer in enumerate(self.hiddenLayers):
            for node in layer:
                if l == 0:
                    n = len(self.inputLayer)
                else:
                    n = len(self.hiddenLayers[l-1])
                node.initialWeight(n)
                for w in node.w:
                    self.x.append(w)

        self.x = np.array(self.x)
        self.v = np.random.uniform(-0.5, 0.5,
                                   size=len(self.x)).reshape(len(self.x), 1)

    def updateNeuralNetwork(self):
        count = 0
        for node in self.outputLayer:
            for i in range(len(node.w)):
                node.w[i] = self.x[count]
                count = count+1
        for layer in self.hiddenLayers:
            for node in layer:
                for i in range(len(node.w)):
                    node.w[i] = self.x[count]
                    count = count+1

    def evaluate(self, data):
        predict = 120
        MAE = []
        for n, train in enumerate(data):
            if n + predict <= len(data) - 1:
                # print(n)
                # print(train['date'], train['time'])
                # print(data[n+predict]['date'], data[n+predict]['time'])
                # print()
                self.feed_forward(train['input'])
                for node in self.outputLayer:
                    MAE.append(
                        np.abs(node.y - data[n+predict]['desire_output']))
        mean_absolute_err = np.average(MAE)
        return mean_absolute_err
