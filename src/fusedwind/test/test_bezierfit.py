
import unittest
import numpy as np

from fusedwind.lib.bezier import BezierCurve, FitBezier

b=BezierCurve()
b.add_control_point(np.array([0,0]))
b.add_control_point(np.array([0.,-0.2]))
b.add_control_point(np.array([.25,-.2]))
b.add_control_point(np.array([.75,0]))
b.add_control_point(np.array([1,0]))
b.ni = 20
b.update()


class TestConfigureTurbine(unittest.TestCase):

    def test_fix_x(self):

        f = FitBezier()
        f.curve_in = b
        f.CPs = np.array([[0, 0., 0.25, 0.75, 1.], np.zeros(5)]).T
        f.fix_x = True
        f.execute()

        self.assertEqual(np.testing.assert_array_almost_equal(f.curve_out.CPs, b.CPs, decimal=6), None)

    def test_free_x(self):

        f = FitBezier()
        f.curve_in = b
        f.CPs = np.array([[0, 0., 0.25, 0.75, 1.], np.zeros(5)]).T
        f.fix_x = False
        f.execute()

        self.assertEqual(np.testing.assert_array_almost_equal(f.curve_out.CPs, b.CPs, decimal=6), None)

    def test_nCP(self):

        f = FitBezier()
        f.curve_in = b
        f.nCPs = 5
        f.execute()

        self.assertEqual(np.testing.assert_array_almost_equal(f.curve_out.CPs, b.CPs, decimal=6), None)

if __name__ == '__main__':

    unittest.main()
