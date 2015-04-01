
# --- 1 ---

import numpy as np

from openmdao.main.api import Assembly, Component
from openmdao.lib.drivers.api import SimpleCaseIterDriver
from openmdao.lib.datatypes.api import List, VarTree

from fusedwind.interface import implement_base
from fusedwind.turbine.geometry import read_blade_planform
from fusedwind.turbine.geometry_vt import BladePlanformVT
from fusedwind.turbine.configurations import configure_bladestructure,\
                                             configure_bladesurface
from fusedwind.turbine.blade_structure import SplinedBladeStructure
from fusedwind.turbine.structure_vt import CrossSectionStructureVT, BeamStructureVT
from fusedwind.turbine.turbine_vt import AeroelasticHAWTVT, configure_turbine

from fusedwind.turbine.aeroelastic_solver import AeroElasticSolverBase
from fusedwind.turbine.blade_structure import BeamStructureCSCode

from fusedwind.turbine.environment_vt import TurbineEnvironmentVT
from fusedwind.turbine.rotoraero_vt import RotorOperationalData, \
                                           DistributedLoadsExtVT, \
                                           RotorLoadsVT, \
                                           BeamDisplacementsVT
from fusedwind.lib.utils import init_vartree



class CS2Dsolver(Component):

    cs2d = VarTree(CrossSectionStructureVT(), iotype='in')
    csprops = VarTree(BeamStructureVT(), iotype='out', desc='1-D Structural beam properties')

    def __init__(self):
        super(CS2Dsolver, self).__init__()

        self.csprops = init_vartree(self.csprops, 1)

    def execute(self):
        
        self.csprops = self.csprops.copy()

        for i, name in enumerate(self.csprops.list_vars()):
            setattr(self.csprops, name, np.array([i + self.cs2d.s]))

        self.csprops.s = np.array(self.cs2d.s).copy()


class BeamStructurePost(Component):

    csprops = List(iotype='in')
    beam_structure = VarTree(BeamStructureVT(), iotype='out', desc='Structural beam properties')

    def execute(self):

        self.beam_structure = self.beam_structure.copy()

        ni = len(self.csprops)
        
        for name in self.beam_structure.list_vars():
            var = getattr(self.beam_structure, name)

            if isinstance(var, np.ndarray):
                var = np.zeros(ni)
                for i, h2d in enumerate(self.csprops):
                    try:
                        var[i] = getattr(h2d, name)
                    except:
                        pass

            setattr(self.beam_structure, name, var)


@implement_base(BeamStructureCSCode)
class BeamStructureAsym(Assembly):

    cs2d = List(CrossSectionStructureVT, iotype='in', desc='Blade cross sectional structure geometry')
    pf = VarTree(BladePlanformVT(), iotype='in', desc='Blade planform discretized according to'
                                                      'the structural resolution')

    beam_structure = VarTree(BeamStructureVT(), iotype='out', desc='Structural beam properties')

    def configure(self):

        self.add('cid', SimpleCaseIterDriver())
        self.driver.workflow.add('cid')

        self.add('cs_code', CS2Dsolver())
        self.cid.workflow.add('cs_code')

        self.add('post', BeamStructurePost())
        self.driver.workflow.add('post')

        self.cid.add_parameter('cs_code.cs2d')
        self.cid.add_response('cs_code.csprops')

        self.connect('cs2d', 'cid.case_inputs.cs_code.cs2d')
        self.connect('cid.case_outputs.cs_code.csprops', 'post.csprops')
        self.connect('post.beam_structure', 'beam_structure')

# --- 2 ---

top = Assembly()

pfIn = read_blade_planform('data/DTU_10MW_RWT_blade_axis_prebend.dat')
configure_bladesurface(top, pfIn, planform_nC=6, span_ni=50, chord_ni=200)
configure_bladestructure(top, 'data/DTU10MW', structure_nC=5, structure_ni=12)

top.st_writer.filebase = 'st_test'

top.blade_length = 86.366

for f in ['data/ffaw3241.dat',
          'data/ffaw3301.dat',
          'data/ffaw3360.dat',
          'data/ffaw3480.dat',
          'data/tc72.dat',
          'data/cylinder.dat']:

    top.blade_surface.base_airfoils.append(np.loadtxt(f))

top.blade_surface.blend_var = np.array([0.241, 0.301, 0.36, 0.48, 0.72, 1.])

# spanwise distribution of planform spline DVs
top.pf_splines.Cx = [0, 0.2, 0.4, 0.6, 0.8, 1.]

# spanwise distribution of sptructural spline DVs
top.st_splines.Cx = [0, 0.2, 0.4, 0.75, 1.]

# spanwise distribution of points where
# cross-sectional structure vartrees will be created
top.st_x = np.linspace(0, 1, 12)

# --- 3 ---

# add structural solver
top.add('st', BeamStructureAsym())
top.driver.workflow.add('st')

top.st.post.beam_structure = init_vartree(top.st.post.beam_structure, top.st_x.shape[0])
top.st.beam_structure = init_vartree(top.st.beam_structure, top.st_x.shape[0])


# --- 5 ---

# connect the parameterized structure vartrees to the 
# structural solver
top.connect('st_builder.cs2d', 'st.cs2d')

# --- 6 ---


import unittest

class BladeSurfaceTestCase(unittest.TestCase):

    def test_cid(self):

        top.run()
        ni = top.st.beam_structure.s.shape[0]
        for i, name in enumerate(top.st.beam_structure.list_vars()):
            if name == 's': continue
            s = top.st.beam_structure.s
            var = getattr(top.st.beam_structure, name)
            self.assertEqual(np.testing.assert_almost_equal(var - s, np.array([i]*ni)), None)


if __name__ == '__main__':

    unittest.main()
    # top.run()


