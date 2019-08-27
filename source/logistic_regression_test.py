"""单元测试代码
测试 logistic_regression.py
"""

import unittest
from logistic_regression import *


class Test_Logistic_Regression(unittest.TestCase):
    def test_sigmoid_with_scalar(self):
        z = 0
        expect = 0.5
        result = sigmoid(z)

        self.assertEqual(expect, result)

    def test_sigmoid_with_vector(self):
        z = np.full(5, 0)
        expect = np.full(5, 0.5)
        result = sigmoid(z)

        self.assertTrue(np.array_equal(expect, result))

    def test_sigmoid_with_matrix(self):
        z = np.full((3, 4), 0)
        expect = np.full((3, 4), 0.5)
        result = sigmoid(z)

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
        result_J, result_gradient = cost_function(theta, X, y)
        result_J = round(result_J, 6)
        result_gradient = np.round(result_gradient, 6)

        self.assertEqual(expect_J, result_J)
        self.assertTrue(np.array_equal(expect_gradient, result_gradient))


if __name__ == '__main__':
    unittest.main()