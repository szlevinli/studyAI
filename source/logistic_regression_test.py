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

if __name__ == '__main__':
    unittest.main()
