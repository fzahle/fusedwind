
from fusedwind.interface import base, implement_base
from fusedwind.lib.utils import init_vartree


def configure_planform(cls, pfIn, planform_nC=8, span_ni=50, spline_type='bezier'):
    """
    method that adds a ``SplinedBladePlanform`` instance to the assembly

    Parameters
    ----------
    cls: class instance
        Instance of an OpenMDAO Assembly that the analysis is run from
    planform_nC: int
        number of spline control points for the planform variables
    span_ni: int
        number of points for the output planform vartree 
    """

    from fusedwind.turbine.geometry import SplinedBladePlanform

    cls.add('pf_splines', SplinedBladePlanform())
    cls.driver.workflow.add('pf_splines')
    cls.pf_splines.nC = planform_nC
    cls.pf_splines.pfIn = pfIn

    cls.create_passthrough('pf_splines.blade_length')
    cls.create_passthrough('pf_splines.span_ni')
    cls.span_ni = span_ni
    cls.pf_splines.configure_splines(spline_type)


def configure_bladesurface(cls, pfIn, planform_nC=8, span_ni=50, chord_ni=300, spline_type='bezier'):
    """
    method that adds a ``LoftedBladeSurface`` instance to the assembly

    Parameters
    ----------
    cls: class instance
        Instance of an OpenMDAO Assembly that the analysis is run from
    planform_nC: int
        number of spline control points for the planform variables
    span_ni: int
        number of points for the output planform vartree
    """

    from fusedwind.turbine.geometry import LoftedBladeSurface

    if not hasattr(cls, 'pf_splines'):
        configure_planform(cls, pfIn, planform_nC, span_ni, spline_type=spline_type)

    cls.add('blade_surface', LoftedBladeSurface(chord_ni, span_ni))
    cls.driver.workflow.add('blade_surface')

    cls.connect('pf_splines.pfOut', 'blade_surface.pf')
    cls.connect('span_ni', 'blade_surface.span_ni')


def configure_bladestructure(cls, file_base, structure_nC=8, structure_ni=12):
    """
    method for configuring an assembly with 
    blade geometry and structural parameterization
    of a blade.

    Parameters
    ----------
    cls: class instance
        Instance of an OpenMDAO Assembly that the analysis is run from
    file_base: str
        path + filebase to the blade structure files, e.g. data/DTU10MW
    planform_nC: int
        number of spline control points for the planform variables
    structure_nC: int
        number of spline control points for the structural variables
    """

    if not hasattr(cls, 'blade_surface'):
        raise RuntimeError('blade_surface needs to be configured before\ncalling this method')

    from fusedwind.turbine.blade_structure import BladeStructureReader, \
                                                  BladeStructureWriter, \
                                                  SplinedBladeStructure, \
                                                  BladeStructureCSBuilder
    from fusedwind.turbine.geometry import RedistributedBladePlanform


    cls.add('st_reader', BladeStructureReader())
    cls.add('st_splines', SplinedBladeStructure())
    cls.add('st_builder', BladeStructureCSBuilder())
    cls.add('st_writer', BladeStructureWriter())

    cls.driver.workflow.add(['st_splines',
                             'st_builder', 
                             'st_writer'])
    cls.create_passthrough('st_splines.x', alias='st_x')

    cls.add('st_pf', RedistributedBladePlanform())
    cls.driver.workflow.add('st_pf')
    cls.st_pf.pfOut = init_vartree(cls.st_pf.pfOut, structure_ni)

    cls.connect('st_x', 'st_pf.x')

    cls.connect('pf_splines.pfOut', 'st_pf.pfIn')

    cls.connect('st_pf.pfOut', 'st_splines.pfOut')
    cls.connect('blade_length', 'st_builder.blade_length')
    # connect the blade structure vartrees through the chain
    # of components
    cls.connect('st_splines.st3dOut', 'st_builder.st3d')
    cls.connect('st_splines.st3dOut', 'st_writer.st3d')

    # connect the stacked blade surface to the st_builder component
    cls.connect('blade_surface.surfnorot', 'st_builder.surface')

    cls.st_reader.filebase = file_base
    cls.st_reader.execute()

    cls.st_splines.add('st3dIn', cls.st_reader.st3d.copy())
    cls.st_splines.nC = structure_nC
    cls.st_splines.span_ni = structure_ni
    cls.st_splines.configure_bladestructure()
    cls.st_builder.add('st3d', cls.st_splines.st3dOut.copy())
    cls.st_builder.add('surface', cls.blade_surface.surfnorot.copy())
    cls.st_builder.generate_cs2d(cls.st_reader.st3d)
    cls.st_writer.add('st3d', cls.st_splines.st3dOut.copy())
