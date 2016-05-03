import numpy as np


def sigmoid(z, alpha=1.0, theta=0.0):
    return 1.0 / (1.0 + np.exp(-alpha * z) * np.exp(-theta))


class Neuron:
    def __init__(self, w, epsilon=0.5):
        self._w = w
        self._epsilon = epsilon

    def get_output(self, x):
        return sigmoid(self._w.dot(x))

    def learn(self, teaching_datum):
        """
        :param teaching_datum: [data0, data1, data2, ..., answer]
        """
        x = teaching_datum[:-1]
        y = teaching_datum[-1]
        val = self.get_output(x)
        self._w -= self._epsilon * 2.0 * (val - y) * val * (1.0 - val) * x


if __name__ == '__main__':
    datum = np.array([0.0, 1.0])
    teaching_data = np.array([0.0, 1.0, 1.0])
    w = 2 * np.random.random(2) - 1.0
    neuron = Neuron(w)

    print('Before learning: ', neuron.get_output(datum))
    for i in range(0, 100):
        neuron.learn(teaching_data)
    print('After learning:', neuron.get_output(datum))
