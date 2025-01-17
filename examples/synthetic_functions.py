from __future__ import division
from builtins import range
from builtins import object
from past.utils import old_div
import numpy
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
import math


class Branin(object):
    def __init__(self):
        self._dim = 2
        self._search_domain = numpy.array([[0.0, 15.0], [-5.0, 15.0]])
        self._num_init_pts = 3
        self._sample_var = 0.0
        self._min_value = 0.397887
        self._observations = []  # numpy.arange(self._dim)
        self._num_fidelity = 0

    def evaluate_true(self, x):
        """ This function is usually evaluated on the square x_1 \in [0, 15], x_2 \in [-5, 15]. Global minimum
        is at x = [pi, 2.275] and [9.42478, 2.475] with minima f(x*) = 0.397887.

            :param x[2]: 2-dim numpy array
        """
        a = 1
        b = old_div(5.1, (4 * pow(numpy.pi, 2.0)))
        c = old_div(5, numpy.pi)
        r = 6
        s = 10
        t = old_div(1, (8 * numpy.pi))
        return numpy.array(
            [(a * pow(x[1] - b * pow(x[0], 2.0) + c * x[0] - r, 2.0) + s * (1 - t) * numpy.cos(x[0]) + s),
             (2 * a * (x[1] - b * pow(x[0], 2.0) + c * x[0] - r) * (-2 * b * x[0] + c) + s * (1 - t) * (
                 -numpy.sin(x[0]))),
             (2 * a * (x[1] - b * pow(x[0], 2.0) + c * x[0] - r))])

    def evaluate(self, x):
        return self.evaluate_true(x)


class Rosenbrock(object):
    def __init__(self):
        self._dim = 20
        self._search_domain = numpy.repeat([[-2., 2.]], self._dim, axis=0)
        self._num_init_pts = 3
        self._sample_var = 0.0
        self._min_value = 0.0
        self._observations = []
        self._num_fidelity = 0

    def evaluate_true(self, x):
        """ Global minimum is 0 at (1, 1, 1, 1)

            :param x[4]: 4-dimension numpy array
        """
        value = 0.0
        for i in range(self._dim - 1):
            value += pow(1. - x[i], 2.0) + 100. * pow(x[i + 1] - pow(x[i], 2.0), 2.0)
        results = [value]
        for i in range(self._dim - 1):
            results += [(2. * (x[i] - 1) - 400. * x[i] * (x[i + 1] - pow(x[i], 2.0)))]
        results += [(200. * (x[self._dim - 1] - pow(x[self._dim - 2], 2.0)))]
        return numpy.array(results)

    def evaluate(self, x):
        return self.evaluate_true(x)


class Hartmann3(object):
    def __init__(self):
        self._dim = 3
        self._search_domain = numpy.repeat([[0., 1.]], self._dim, axis=0)
        self._num_init_pts = 3
        self._sample_var = 0.0
        self._min_value = -3.86278
        self._observations = []  # numpy.arange(self._dim)
        self._num_fidelity = 0

    def evaluate_true(self, x):
        """ domain is x_i \in (0, 1) for i = 1, ..., 3
            Global minimum is -3.86278 at (0.114614, 0.555649, 0.852547)

            :param x[3]: 3-dimension numpy array with domain stated above
        """
        alpha = numpy.array([1.0, 1.2, 3.0, 3.2])
        A = numpy.array([[3., 10., 30.], [0.1, 10., 35.], [3., 10., 30.], [0.1, 10., 35.]])
        P = 1e-4 * numpy.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])
        results = [0.0] * 4
        for i in range(4):
            inner_value = 0.0
            for j in range(self._dim):
                inner_value -= A[i, j] * pow(x[j] - P[i, j], 2.0)
            results[0] -= alpha[i] * numpy.exp(inner_value)
            for j in range(self._dim - self._num_fidelity):
                results[j + 1] -= (alpha[i] * numpy.exp(inner_value)) * ((-2) * A[i, j] * (x[j] - P[i, j]))
        return numpy.array(results)

    def evaluate(self, x):
        t = self.evaluate_true(x)
        return t


class Levy4(object):
    def __init__(self):
        self._dim = 4
        self._search_domain = numpy.repeat([[-5., 5.]], self._dim, axis=0)
        self._num_init_pts = 3
        self._sample_var = 0.0
        self._min_value = 0.0
        self._observations = []  # numpy.arange(self._dim)
        self._num_fidelity = 0

    def evaluate_true(self, x):
        """ Global minimum is 0 at (1, 1, 1, 1)

            :param x[4]: 4-dimension numpy array

            a difficult test case for KG-type methods.
        """
        x = numpy.asarray_chkfinite(x)
        n = len(x)
        z = 1 + old_div((x - 1), 4)

        results = [0] * (n + 1)
        results[0] = (sin(pi * z[0]) ** 2
                      + sum((z[:-1] - 1) ** 2 * (1 + 10 * sin(pi * z[:-1] + 1) ** 2))
                      + (z[-1] - 1) ** 2 * (1 + sin(2 * pi * z[-1]) ** 2))
        results[1] = 2. * sin(pi * z[0]) * cos(pi * z[0]) * pi * (0.25)
        results[n] = (((z[-1] - 1) ** 2) * (2. * sin(2 * pi * z[-1]) * cos(2. * pi * z[-1]) * 2. * pi * (0.25))
                      + 2. * (z[-1] - 1) * (0.25) * (1 + sin(2. * pi * z[-1]) ** 2))

        results[1:-1] += (((z[:-1] - 1) ** 2) * (20. * sin(pi * z[:-1] + 1) * cos(pi * z[:-1] + 1) * pi * (0.25))
                          + 2 * (z[:-1] - 1) * (0.25) * (1 + 10. * sin(pi * z[:-1] + 1) ** 2))
        return numpy.array(results)

    def evaluate(self, x):
        t = self.evaluate_true(x)
        return t


class Hartmann6(object):
    def __init__(self):
        self._dim = 6
        self._search_domain = numpy.repeat([[0., 1.]], self._dim, axis=0)
        self._num_init_pts = 3
        self._sample_var = 0.0
        self._min_value = -3.32237
        self._observations = []  # numpy.arange(self._dim)
        self._num_fidelity = 0

    def evaluate_true(self, x):
        """ domain is x_i \in (0, 1) for i = 1, ..., 6
            Global minimum is -3.32237 at (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)

            :param x[6]: 6-dimension numpy array with domain stated above
        """
        alpha = numpy.array([1.0, 1.2, 3.0, 3.2])
        A = numpy.array([[10, 3, 17, 3.50, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8],
                         [17, 8, 0.05, 10, 0.1, 14]])
        P = 1.0e-4 * numpy.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
                                  [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])
        results = [0.0] * 7
        for i in range(4):
            inner_value = 0.0
            for j in range(self._dim - self._num_fidelity):
                inner_value -= A[i, j] * pow(x[j] - P[i, j], 2.0)
            results[0] -= alpha[i] * numpy.exp(inner_value)
            for j in range(self._dim - self._num_fidelity):
                results[j + 1] -= (alpha[i] * numpy.exp(inner_value)) * ((-2) * A[i, j] * (x[j] - P[i, j]))
        return numpy.array(results)

    def evaluate(self, x):
        return self.evaluate_true(x)


class Ackley(object):
    def __init__(self):
        self._dim = 10
        self._search_domain = numpy.repeat([[-1., 1.]], self._dim, axis=0)
        self._num_init_pts = 3
        self._sample_var = 0.0
        self._min_value = 0.0
        self._observations = []
        self._num_fidelity = 0

    def evaluate_true(self, x):
        x = 20 * x
        firstSum = 0.0
        secondSum = 0.0
        for c in x:
            firstSum += c ** 2.0
            secondSum += math.cos(2.0 * math.pi * c)
        n = float(len(x))
        results = [old_div(
            (-20.0 * math.exp(-0.2 * math.sqrt(old_div(firstSum, n))) - math.exp(old_div(secondSum, n)) + 20 + math.e),
            6.)]
        for i in range(int(n)):
            results += [-20.0 * math.exp(-0.2 * math.sqrt(old_div(firstSum, n))) * (
                        -0.2 * (old_div(x[i], n)) / (math.sqrt(old_div(firstSum, n)))) -
                        math.exp(old_div(secondSum, n)) * (2.0 * math.pi / n) * (-math.sin(2.0 * math.pi * x[i]))]

        return numpy.array(results)

    def evaluate(self, x):
        t = self.evaluate_true(x)
        results = []
        for r in t:
            n = numpy.random.normal(0, numpy.sqrt(self._sample_var))
            results += [r + n]
        return numpy.array(results)


class Ackley20(object):
    def __init__(self):
        self._dim = 20
        self._search_domain = numpy.repeat([[-1., 1.]], self._dim, axis=0)
        self._num_init_pts = 3
        self._sample_var = 0.0
        self._min_value = 0.0
        self._observations = []
        self._num_fidelity = 0

    def evaluate_true(self, x):
        x = 20 * x
        firstSum = 0.0
        secondSum = 0.0
        for c in x:
            firstSum += c ** 2.0
            secondSum += math.cos(2.0 * math.pi * c)
        n = float(len(x))
        results = [old_div(
            (-20.0 * math.exp(-0.2 * math.sqrt(old_div(firstSum, n))) - math.exp(old_div(secondSum, n)) + 20 + math.e),
            6.)]
        for i in range(int(n)):
            results += [-20.0 * math.exp(-0.2 * math.sqrt(old_div(firstSum, n))) * (
                        -0.2 * (old_div(x[i], n)) / (math.sqrt(old_div(firstSum, n)))) -
                        math.exp(old_div(secondSum, n)) * (2.0 * math.pi / n) * (-math.sin(2.0 * math.pi * x[i]))]

        return numpy.array(results)

    def evaluate(self, x):
        t = self.evaluate_true(x)
        results = []
        for r in t:
            n = numpy.random.normal(0, numpy.sqrt(self._sample_var))
            results += [r + n]
        return numpy.array(results)
