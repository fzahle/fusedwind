import numpy as np
from math import pi
import copy
from openmdao.main.api import VariableTree
from openmdao.lib.datatypes.api import Str, VarTree, Float
#from fusedwind.turbine.turbine_vt import AeroElasticSimulationSetup, TurbineEnvironmentVT, OffshoreTurbineEnvironmentVT
from fusedwind.turbine.environment_vt import  TurbineEnvironmentVT, OffshoreTurbineEnvironmentVT
from fusedwind.lib.base import CaseInputsBase, CaseOutputsBase

class IECRunCaseBaseVT(VariableTree):

    case_name = Str('IEC_case', desc='Name of the specific case passed to the aeroelastic code')
# This appears to be broken
#    simulation = VarTree(AeroElasticSimulationSetup(), desc='Basic simulation input settings')
    environment = VarTree(TurbineEnvironmentVT(), desc='Inflow conditions to the turbine simulation')

class IECOutputsBaseVT(VariableTree):

    pmek = Float()
    thrust = Float()

###### Peter's stuff #######################

class RunCaseBuilder(object):
    """ base class for run case builders.  only one we have is FAST 
    single aerodcode run.
    Anything in common to build toward here ??? 
    We want to push anything that is not specific to a certain aerocode to be set up here """
    pass


#//////////////////////////////////////////////////

class GenericRunCase(IECRunCaseBaseVT):
    """ Like Run case, but only base key value info
    still one run of aero code, just w.r.t. "universal" variables """
    def __init__(self,casename, param_names, ln):
        super(GenericRunCase,self).__init__()
        self.x = np.array(ln)
        self.param_names = param_names
        self.sample = {param_names[i]:self.x[i] for i in range(len(param_names))}
        self.thename = casename
        for p in self.sample:
            self.thename += "%s.%.1f" % (p[0:3],self.sample[p])
        self.case_name = self.thename

class GenericRunCaseTable(object):
    """ basically a list of GenericRunCase's', including header"""
    def initFromFile(self, filename, start_at = 0, verbose =True):
        ## special "table"-like file: one line header of variable names then data only
        print "reading raw cases from ", filename
        fin = file(filename).readlines()
#       print "lines: ", fin
        casename = "raw_cases"
        self.cases = []
        param_names = fin[0].split()
        print "PNAMES",param_names
        for iln in range(1,len(fin)):
            if (iln > start_at):  # beware indexing: 0th sample is on line 1, after header, so we need line idex > start_at
                ln = fin[iln].strip().split()
#            print "parsing:", ln
                ln = [float(f) for f in ln]
                self.cases.append(GenericRunCase(casename, param_names, ln))
            

#///////////////////////////////////////////

class RunCase(GenericRunCase):
    """ base class for specific run case (one run) of a specific aero code"""
    def __init__(self, case, generic_sample):        
        super(RunCase,self).__init__(case, generic_sample.keys(), generic_sample.values())
        self.case_name = case
        self.sample = generic_sample


#///////////////////////////////////////////////////

class RunResult(object):
    """ class to modularize collecting the results of all the runs, may have to use the
    Dispatcher and AeroCode to find and parse them, resp."""
    ## Have yet to do anything useful with this class
    # so far I am just copying back my results and directly parsing them (see collect_output)
    ##
    def __init__(self, aerocode):
        self.aerocode = aerocode
    
