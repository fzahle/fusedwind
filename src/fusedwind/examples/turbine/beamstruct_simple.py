
# --- 1 ---
import time
import numpy as np

from openmdao.main.mpiwrap import MPI
from openmdao.main.api import Assembly, Component, set_as_top
from openmdao.lib.drivers.api import SimpleCaseIterDriver
from openmdao.lib.drivers.mpicasedriver import MPICaseDriver
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

        time.sleep(1.)
        self.csprops = self.csprops.copy()
        try:
            print 'CS2Dsolver: s=%f, rank %i, sleeping 1 sec' % (self.cs2d.s, MPI.COMM_WORLD.rank)
        except:
            print 'executing serial', self.cs2d.s
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

class BeamStructureAsymFail(Assembly):

    # cs2d = List(CrossSectionStructureVT, iotype='in', desc='Blade cross sectional structure geometry')
    # pf = VarTree(BladePlanformVT(), iotype='in', desc='Blade planform discretized according to'
    #                                                   'the structural resolution')

    # beam_structure = VarTree(BeamStructureVT(), iotype='out', desc='Structural beam properties')

    def configure(self):

        self.driver.system_type = 'serial'
        # for the CID to work in parallel, it has to be added as 'driver'
        # self.add('driver', MPICaseDriver())
        self.add('cid', MPICaseDriver())
        # self.add('cid', SimpleCaseIterDriver())
        self.driver.workflow.add('cid')

        self.add('cs_code', CS2Dsolver())
        # self.driver.workflow.add('cs_code')
        # doesn't work
        self.cid.workflow.add('cs_code')

        # doesn't work
        self.add('post', BeamStructurePost())
        self.driver.workflow.add('post')

        # self.driver.add_parameter('cs_code.cs2d')
        # self.driver.add_response('cs_code.csprops')
        self.cid.add_parameter('cs_code.cs2d')
        self.cid.add_response('cs_code.csprops')

        # passthorughs don't work under MPI
        # self.create_passthrough('cid.case_inputs.cs_code.cs2d')
        self.create_passthrough('post.beam_structure')
        self.connect('cid.case_outputs.cs_code.csprops', 'post.csprops')

# @implement_base(BeamStructureCSCode)
class BeamStructureAsym(Assembly):

    # cs2d = List(CrossSectionStructureVT, iotype='in', desc='Blade cross sectional structure geometry')
    # pf = VarTree(BladePlanformVT(), iotype='in', desc='Blade planform discretized according to'
    #                                                   'the structural resolution')

    # beam_structure = VarTree(BeamStructureVT(), iotype='out', desc='Structural beam properties')

    def configure(self):

        # for the CID to work in parallel, it has to be added as 'driver'
        self.add('driver', MPICaseDriver())
        # self.add('cid', MPICaseDriver())
        # self.add('cid', SimpleCaseIterDriver())
        # self.driver.workflow.add('cid')

        self.add('cs_code', CS2Dsolver())
        self.driver.workflow.add('cs_code')
        # doesn't work
        # self.cid.workflow.add('cs_code')

        # doesn't work
        # self.add('post', BeamStructurePost())
        # self.driver.workflow.add('post')

        self.driver.add_parameter('cs_code.cs2d')
        self.driver.add_response('cs_code.csprops')
        # self.driver.add_parameter('cs_code.cs2d')
        # self.driver.add_response('cs_code.csprops')

        # passthorughs don't work
        # self.create_passthrough('driver.case_inputs.cs_code.cs2d')
        # self.connect('cid.case_outputs.cs_code.csprops', 'post.csprops')
        # self.connect('post.beam_structure', 'beam_structure')

# --- 2 ---

class AnalysisFail(Assembly):
    """
    this analysis fails due the passthroughs and combination
    of simple driver and CID in one assembly.
    """

    def configure(self):

        # add structural solver
        self.add('st', BeamStructureAsymFail())
        self.driver.workflow.add('st')

        for i in range(8):
            cs = CrossSectionStructureVT()
            cs.s = i
            self.st.cid.case_inputs.cs_code.cs2d.append(cs)
            # self.st.cs2d.append(cs)
        # self.st.post.beam_structure = init_vartree(self.st.post.beam_structure, 8)
        # self.st.beam_structure = init_vartree(self.st.beam_structure, 8)

class Analysis(Assembly):
    """
    this stripped down version with no passthoughs works
    """

    def __init__(self):
        super(Analysis, self).__init__()

        self.driver.system_type = 'serial'

    def configure(self):


        pfIn = read_blade_planform('data/DTU_10MW_RWT_blade_axis_prebend.dat')
        configure_bladesurface(self, pfIn, planform_nC=6, span_ni=50, chord_ni=200)
        configure_bladestructure(self, 'data/DTU10MW', structure_nC=5, structure_ni=12)

        self.st_writer.filebase = 'st_test'

        self.blade_length = 86.366

        for f in ['data/ffaw3241.dat',
                  'data/ffaw3301.dat',
                  'data/ffaw3360.dat',
                  'data/ffaw3480.dat',
                  'data/tc72.dat',
                  'data/cylinder.dat']:

            self.blade_surface.base_airfoils.append(np.loadtxt(f))

        self.blade_surface.blend_var = np.array([0.241, 0.301, 0.36, 0.48, 0.72, 1.])

        # spanwise distribution of planform spline DVs
        self.pf_splines.Cx = [0, 0.2, 0.4, 0.6, 0.8, 1.]

        # spanwise distribution of sptructural spline DVs
        self.st_splines.Cx = [0, 0.2, 0.4, 0.75, 1.]

        # spanwise distribution of points where
        # cross-sectional structure vartrees will be created
        self.st_x = np.linspace(0, 1, 12)

        # add structural solver
        self.add('st', BeamStructureAsym())
        self.driver.workflow.add('st')

        # Adding a passthrough in the BeamStructureAsym causes a crash
        # so we just add vartrees directly to the
        # self.st.driver.case_inputs.cs_code.cs2d list
        # self.connect('st_builder.cs2d', 'st.cs2d')

        for i in range(8):
            cs = CrossSectionStructureVT()
            cs.s = i
            self.st.driver.case_inputs.cs_code.cs2d.append(cs)




if __name__ == '__main__':

    # unittest.main()
    # top = AnalysisFail()
    top = Analysis()
    t0 = time.time()
    top.run()
    t1 = time.time() - t0
    t_all = MPI.COMM_WORLD.allgather(t1)
    if MPI.COMM_WORLD.rank == 0:
        print 'mean time: ', np.asarray(t_all).mean()
        print 'total time: ', np.asarray(t_all).sum()
        print 'efficiency', 8.1 / np.asarray(t_all).sum()


