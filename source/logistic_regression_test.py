"""单元测试代码
测试 logistic_regression.py
"""

import unittest
from unittest.mock import MagicMock
import numpy as np
import logistic_regression as lr


class Test_Logistic_Regression(unittest.TestCase):
    def test_sigmoid_with_scalar(self):
        z = 0
        expect = 0.5
        result = lr.sigmoid(z)

        self.assertEqual(expect, result)

    def test_sigmoid_with_vector(self):
        z = np.full(5, 0)
        expect = np.full(5, 0.5)
        result = lr.sigmoid(z)

        self.assertTrue(np.array_equal(expect, result))

    def test_sigmoid_with_matrix(self):
        z = np.full((3, 4), 0)
        expect = np.full((3, 4), 0.5)
        result = lr.sigmoid(z)

        self.assertTrue(np.array_equal(expect, result))

    def test_cost_function(self):
        theta = np.full(5, 0)
        X = np.array([
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
            [30, 31, 32, 33, 34],
            [40, 41, 42, 43, 44],
            [50, 51, 52, 53, 54],
            [60, 61, 62, 63, 64]
        ])
        y = np.reshape(range(100, 106), 6)
        expect_J = 0.693147
        expect_gradient = np.array([
            -3599.166667,
            -3701.166667,
            -3803.166667,
            -3905.166667,
            -4007.166667
        ])
        result_J, result_gradient = lr.cost_function(theta, X, y)
        result_J = round(result_J, 6)
        result_gradient = np.round(result_gradient, 6)

        self.assertEqual(expect_J, result_J)
        self.assertTrue(np.array_equal(expect_gradient, result_gradient))

    def test_map_featrue(self):
        X1 = np.reshape(range(1, 4), (3, 1))
        X2 = np.reshape(range(4, 7), (3, 1))

        expect = np.array([
            [1, 1, 4, 1, 4, 16, 1, 4, 16, 64],
            [1, 2, 5, 4, 10, 25, 8, 20, 50, 125],
            [1, 3, 6, 9, 18, 36, 27, 54, 108, 216]
        ])
        result = lr.map_featrue(X1, X2, 3)

        self.assertTrue(np.array_equal(expect, result))

    def test_cost_function_reg(self):
        theta = np.full(5, 0.3)
        X = np.array([
            [0.01, 0.02, 0.03, 0.04, 0.05],
            [0.02, 0.03, 0.04, 0.05, 0.06],
            [0.03, 0.04, 0.05, 0.06, 0.07],
            [0.04, 0.05, 0.06, 0.07, 0.08],
            [0.05, 0.06, 0.07, 0.08, 0.09],
            [0.06, 0.07, 0.08, 0.09, 0.1]
        ])
        y = np.reshape(range(100, 106), 6)
        l = 2.3
        expect_J = -7.68187

        result_J = lr.cost_function_reg(theta, X, y, l)
        result_J = round(result_J, 5)

        self.assertEqual(expect_J, result_J)

    def test_gradient_reg(self):
        theta = np.full(5, 0.3)
        X = np.array([
            [0.01, 0.02, 0.03, 0.04, 0.05],
            [0.02, 0.03, 0.04, 0.05, 0.06],
            [0.03, 0.04, 0.05, 0.06, 0.07],
            [0.04, 0.05, 0.06, 0.07, 0.08],
            [0.05, 0.06, 0.07, 0.08, 0.09],
            [0.06, 0.07, 0.08, 0.09, 0.1]
        ])
        y = np.reshape(range(100, 106), 6)
        l = 2.3
        expect_gradient = np.array([
            -3.59833614,
            -4.503130041,
            -5.522923941,
            -6.542717842,
            -7.562511743
        ])
        result_gradient = lr.gradient_reg(theta, X, y, l)
        result_gradient = np.round(result_gradient, 9)

        self.assertTrue(np.array_equal(expect_gradient, result_gradient))


if __name__ == '__main__':
    unittest.main()
