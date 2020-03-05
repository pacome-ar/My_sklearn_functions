def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Perceptron():
    def __init__(self, eta=0.001, niter=10000, random_state=42, bias=True):
        self.eta=eta
        self.niter=niter
        self.random_state=np.random.RandomState(seed=random_state)
        self.bias = bias
        self.activation = sigmoid
        self.activation_grad = sigmoid_grad

    def _init_weights(self, n):
        self.weights = self.random_state.random(n)

    def _neuron_output(self, X):
        return self.activation(X @ self.weights)

    def _error(self, y, predicted):
        return y - predicted

    def _gradient_learn(self, Xi, yi):
        predicted = self.predict(Xi)
        grad = self.activation_grad(Xi @ self.weights)
        return self.weights + self.eta * self._error(yi, predicted) * grad * Xi

    def _make_bias(self, X):
        if self.bias:
            Xc = np.hstack((X, np.ones((len(X), 1), dtype='int')))
        else:
            Xc = X
        return Xc

    def predict(self, X):
        return self.activation(X @ self.weights)

    def _fit_once(self, X, y):
        for Xi, yi in zip(X, y):
            self.weights = self._gradient_learn(Xi, yi)

    def fit(self, X, y):
        X = self._make_bias(X)
        self._init_weights(X.shape[-1])
        for i in range(self.niter):
            self._fit_once(X, y)
        return self.predict(X)


import copy

class MLP():
    def __init__(self):
        self.layers = []
        self.neurons = {}

    def _add_layer(self, nb, neuron, input_dim=None):
        if len(self.layers) == 0:
            self.input_dim = input_dim
        nbneuron = len(self.neurons)
        for i in range(nb):
            self.neurons[i + nbneuron] = neuron
        self.layers.append(np.arange(nb) + nbneuron)

    def _add_output_neuron(self, neuron):
        self._add_layer(1, neuron)

    def _init_weights(self):
        assert len(self.layers) > 0, 'must be layers before init'
        for i in self.layers[0]:
            self.neurons[i]._init_weights(self.input_dim)
        n = len(self.layers[0])
        for layer in self.layers[1:]:
            for i in layer:
                self.neurons[i]._init_weights(n)
            n = len(layer)

    def _feed_forward(self, x, start=0, stop=None):
        inputs = copy.deepcopy(x)
        if stop is None:
            stop = len(self.layers)
        for layer in self.layers[start:stop]:
            inputs = np.array([self.neurons[i].predict(inputs) for i in layer]).T
        return inputs

    def _error(self, y, predicted):
        return y - predicted



a = MLP()
a._add_layer(9, Perceptron(bias=False), input_dim=X.shape[1])
a._add_layer(6, Perceptron(bias=False))
a._add_layer(3, Perceptron(bias=False))
a._add_layer(7, Perceptron(bias=False))
a._add_output_neuron(Perceptron(bias=False))

a._init_weights()
a.layers

xx = X[0]
target = y[0]
eta = 1

layer_index = len(a.layers)
pred = a._feed_forward(xx)
xi = a._feed_forward(xx, stop=layer_index - 1)
output_neuron = a.neurons[a.layers[-1][0]]
error = output_neuron.activation_grad(output_neuron.weights @ xi) * (pred - target)
weights = output_neuron.weights
new_weights = weights - eta * error * xi

layer_index -= 1

for layer in a.layers[::-1][1:]:
    print(layer)
    for i in layer:
        neuron = a.neurons[i]
        xi = a._feed_forward(xx, stop=layer_index - 1)
        grad = neuron.activation_grad(neuron.weights @ xi)
        print(error.shape, weights.shape, error_temp)
    layer_index -= 1
