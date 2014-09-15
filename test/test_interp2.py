__author__ = 's'

import unittest
import numpy as np
from interp2 import interp2linear
from numpy import nan
from StringIO import StringIO


class MyTestCase(unittest.TestCase):
    #def test_something(self):
    #    self.assertEqual(True, False)


    def test_interp2linear(self):
        I1 = np.array([
            [0.5, 0., 0.],
            [1., 0., 0.],
            [0.5, 0., 0.],
            [0., 1., 0.],
        ])

        idx, idy = np.meshgrid(np.arange(I1.shape[1]), np.arange(I1.shape[0]))

        self.assertTrue(np.array_equal(interp2linear(I1, idx, idy, ), I1))
        self.assertTrue(np.array_equal(interp2linear(I1, idx.ravel(), idy.ravel(), ), I1.ravel()))

        self.assertTrue(np.isclose(
            interp2linear(I1, idx+0.5, idy, ),
            np.array(
                [[ 0.25,  0.,     nan],
                 [ 0.5 ,  0.,     nan],
                 [ 0.25,  0.,     nan],
                 [ 0.5 ,  0.5,    nan]]
            ),
            atol=1e-10,
            equal_nan=True,
        ).all())


        self.assertTrue(np.isclose(
            interp2linear(I1, idx, idy+0.5, ),
            np.loadtxt(StringIO("""
                0.7500         0         0
                0.7500         0         0
                0.2500    0.5000         0
                nan       nan  nan  """
            )),
            atol=1e-10,
            equal_nan=True,
        ).all())

        self.assertTrue(np.isclose(
            interp2linear(I1, idx+0.5, idy+0.5, ),
            np.loadtxt(StringIO("""
                0.3750         0       NaN
                0.3750         0       NaN
                0.3750    0.2500       NaN
                NaN       NaN       NaN  """
            )),
            atol=1e-10,
            equal_nan=True,
        ).all())

        self.assertTrue(np.isclose(
            interp2linear(I1, idx-1.5, idy, ),
            np.loadtxt(StringIO("""
                NaN       NaN    0.2500
                NaN       NaN    0.5000
                NaN       NaN    0.2500
                NaN       NaN    0.5000"""
            )),
            atol=1e-10,
            equal_nan=True,
        ).all())

        self.assertTrue(np.isclose(
            interp2linear(I1, idx-1.5, idy-1.5, ),
            np.loadtxt(StringIO("""
                NaN       NaN       NaN
                NaN       NaN       NaN
                NaN       NaN    0.3750
                NaN       NaN    0.3750"""
            )),
            atol=1e-10,
            equal_nan=True,
        ).all())

        self.assertTrue(np.isclose(
            interp2linear(I1, idx, idy-1.5, extrapval=99),
            np.loadtxt(StringIO("""
                99       99       99
                99       99       99
                0.7500         0         0
                0.7500         0         0"""
            )),
            atol=1e-10,
            equal_nan=True,
        ).all())

        self.assertTrue(np.isclose(
            interp2linear(I1, np.array([1000.3, 1000.4]).T, np.array([1000.5, -1000.6]).T),
            np.loadtxt(StringIO("""
                nan
                nan"""
            )),
            atol=1e-10,
            equal_nan=True,
        ).all())

        self.assertTrue(np.isclose(
            interp2linear(I1, np.array([0.3, 1.4]), np.array([0.5, 0.6]), extrapval=99),
            np.loadtxt(StringIO("""0.5250         0""")),
            atol=1e-10,
            equal_nan=True,
        ).all())
        pass



if __name__ == '__main__':
    unittest.main()
