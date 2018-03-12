import numpy as np
from math import sqrt, pi, exp


class GNB(object):

    def __init__(self):
        self.possible_labels = ['left', 'keep', 'right']
        self.means = {}
        self.stddevs = {}

    def gaussian_prob(self, obs, mu, sig):
        num = (obs - mu) ** 2
        denum = 2 * sig ** 2
        norm = 1 / sqrt(2 * pi * sig ** 2)
        return norm * exp(-num / denum)

    def train(self, data, labels):
        """
        Trains the classifier with N data points and labels.

        INPUTS
        data - array of N observations
          - Each observation is a tuple with 4 values: s, d,
            s_dot and d_dot.
          - Example : [
                [3.5, 0.1, 5.9, -0.02],
                [8.0, -0.3, 3.0, 2.2],
                ...
            ]

        labels - array of N labels
          - Each label is one of "left", "keep", or "right".
        """

        values_by_label = {}

        """the features for each data point are: s, d%4, s_dot, d_dot
        and the total no of features are 4"""

        for lbl in self.possible_labels:
            values_by_label[lbl] = np.empty((4, 0))

        for X, Y in zip(data, labels):
            values_by_label[Y] = np.append(values_by_label[Y], [[X[0]], [(X[1] % 4)], [X[2]], [X[3]]], axis=1)

        """values by label is a dictionary with three keys: left, keep, and right.
        each key points to a list of shape (4, ) and save the features of every data point
        """

        means = {}
        stddevs = {}

        for label in self.possible_labels:
            temp_array = np.array(values_by_label[label])
            means[label] = np.mean(temp_array, axis=1)
            stddevs[label] = np.std(temp_array, axis=1)

        self.means = means
        self.stddevs = stddevs

    def predict_probs(self, obs):

        """
        Private method used to assign a probability to each class.
        """
        probs = {}
        for label in self.possible_labels:

            product = 1.00
            for feature in range(4):
                product *= self.gaussian_prob(obs[feature], self.means[label][feature], self.stddevs[label][feature])

            probs[label] = product

        """probs is a dictionary containing the probabilities of the observation for each class.
        The class labels are the keys with the values being the respective probabilities for that class."""
        return probs

    def predict(self, observation):
        """
        Once trained, this method is called and expected to return
        a predicted behavior for the given observation.

        INPUTS

        observation - a 4 tuple with s, d, s_dot, d_dot.
          - Example: [3.5, 0.1, 8.5, -0.2]

        OUTPUT

        A label representing the best guess of the classifier. Can
        be one of "left", "keep" or "right".
        """
        probs = self.predict_probs(observation)

        best_probability = 0
        best_label = " "
        for label in self.possible_labels:
            if probs[label] > best_probability:
                best_probability = probs[label]
                best_label = label

        return best_label
