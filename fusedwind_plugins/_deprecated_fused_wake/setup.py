#
# This file is autogenerated during plugin quickstart and overwritten during
# plugin makedist. DO NOT CHANGE IT if you plan to use plugin makedist to update 
# the distribution.
#

from setuptools import setup, find_packages

kwargs = {'author': 'Pierre-Elouan Rethore, DTU Wind Energy',
 'author_email': 'pire@dtu.dk',
 'classifiers': ['Intended Audience :: Science/Research',
                 'Topic :: Scientific/Engineering'],
 'description': '',
 'download_url': 'https://github.com/FUSED-Wind/FUSED-Wind/tree/develop/fusedwind_plugins/fused_wake',
 'entry_points': '[openmdao.component]\nfused_wake.wake.WakeDriver=fused_wake.wake:WakeDriver\nfusedwind.plant_flow.fused_plant_comp.GenericWakeModel=fusedwind.plant_flow.fused_plant_comp:GenericWakeModel\nfused_wake.wake.UpstreamWakeDriver=fused_wake.wake:UpstreamWakeDriver\nfusedwind.plant_flow.fused_plant_comp.HubCenterWSPosition=fusedwind.plant_flow.fused_plant_comp:HubCenterWSPosition\nfused_wake.wake.NeutralLogLawInflowGenerator=fused_wake.wake:NeutralLogLawInflowGenerator\nfused_wake.wake.WakeReader=fused_wake.wake:WakeReader\nfused_wake.wake.WTStreamwiseSorting=fused_wake.wake:WTStreamwiseSorting\nfusedwind.plant_flow.fused_plant_comp.WindTurbinePowerCurve=fusedwind.plant_flow.fused_plant_comp:WindTurbinePowerCurve\nfusedwind.plant_flow.fused_plant_comp.GenericWSPosition=fusedwind.plant_flow.fused_plant_comp:GenericWSPosition\nfused_wake.wake.WTID=fused_wake.wake:WTID\nfused_wake.noj.NOJWakeModel=fused_wake.noj:NOJWakeModel\nfused_wake.wake.PostProcessWTCases=fused_wake.wake:PostProcessWTCases\nfused_wake.wake.GenericWindFarmWake=fused_wake.wake:GenericWindFarmWake\nfusedwind.plant_flow.fused_plant_comp.GenericFlowModel=fusedwind.plant_flow.fused_plant_comp:GenericFlowModel\nfused_wake.wake.QuadraticWakeSum=fused_wake.wake:QuadraticWakeSum\nfusedwind.plant_flow.fused_plant_comp.GenericWindTurbine=fusedwind.plant_flow.fused_plant_comp:GenericWindTurbine\nfusedwind.plant_flow.fused_plant_comp.GenericHubWindSpeed=fusedwind.plant_flow.fused_plant_comp:GenericHubWindSpeed\nfused_wake.noj.NOJWindFarmWake=fused_wake.noj:NOJWindFarmWake\nfused_wake.wake.PostProcessWindRose=fused_wake.wake:PostProcessWindRose\nfused_wake.wake.HubCenterWS=fused_wake.wake:HubCenterWS\nfused_wake.wake.HomogeneousInflowGenerator=fused_wake.wake:HomogeneousInflowGenerator\nfused_wake.wake.GenericEngineeringWakeModel=fused_wake.wake:GenericEngineeringWakeModel\nfusedwind.plant_flow.fused_plant_comp.GenericInflowGenerator=fusedwind.plant_flow.fused_plant_comp:GenericInflowGenerator\nfused_wake.wake.GenericWindFarm=fused_wake.wake:GenericWindFarm\nfused_wake.noj.MozaicTileWindFarmWake=fused_wake.noj:MozaicTileWindFarmWake\nfused_wake.noj.MozaicTileWakeModel=fused_wake.noj:MozaicTileWakeModel\nfused_wake.wake.GenericInflowGenerator=fused_wake.wake:GenericInflowGenerator\nfused_wake.wake.GenericAEP=fused_wake.wake:GenericAEP\nfusedwind.plant_flow.fused_plant_comp.GenericWakeSum=fusedwind.plant_flow.fused_plant_comp:GenericWakeSum\nfused_wake.wake.AEP=fused_wake.wake:AEP\nfused_wake.wake.LinearWakeSum=fused_wake.wake:LinearWakeSum\nfusedwind.plant_flow.fused_plant_comp.PostProcessWindRose=fusedwind.plant_flow.fused_plant_comp:PostProcessWindRose\n\n[openmdao.driver]\nfused_wake.wake.WakeDriver=fused_wake.wake:WakeDriver\nfused_wake.wake.UpstreamWakeDriver=fused_wake.wake:UpstreamWakeDriver\n\n[openmdao.container]\nfused_wake.wake.WakeDriver=fused_wake.wake:WakeDriver\nfusedwind.plant_flow.fused_plant_comp.GenericWakeModel=fusedwind.plant_flow.fused_plant_comp:GenericWakeModel\nfused_wake.wake.UpstreamWakeDriver=fused_wake.wake:UpstreamWakeDriver\nfusedwind.plant_flow.fused_plant_comp.HubCenterWSPosition=fusedwind.plant_flow.fused_plant_comp:HubCenterWSPosition\nfused_wake.wake.NeutralLogLawInflowGenerator=fused_wake.wake:NeutralLogLawInflowGenerator\nfused_wake.wake.WakeReader=fused_wake.wake:WakeReader\nfused_wake.wake.WTStreamwiseSorting=fused_wake.wake:WTStreamwiseSorting\nfusedwind.plant_flow.fused_plant_comp.WindTurbinePowerCurve=fusedwind.plant_flow.fused_plant_comp:WindTurbinePowerCurve\nfused_wake.wake.WTID=fused_wake.wake:WTID\nfused_wake.noj.NOJWakeModel=fused_wake.noj:NOJWakeModel\nfused_wake.wake.PostProcessWTCases=fused_wake.wake:PostProcessWTCases\nfused_wake.wake.GenericWindFarmWake=fused_wake.wake:GenericWindFarmWake\nfusedwind.plant_flow.fused_plant_comp.GenericFlowModel=fusedwind.plant_flow.fused_plant_comp:GenericFlowModel\nfusedwind.plant_flow.fused_plant_comp.GenericWSPosition=fusedwind.plant_flow.fused_plant_comp:GenericWSPosition\nfused_wake.wake.QuadraticWakeSum=fused_wake.wake:QuadraticWakeSum\nfusedwind.plant_flow.fused_plant_comp.GenericWindTurbine=fusedwind.plant_flow.fused_plant_comp:GenericWindTurbine\nfusedwind.plant_flow.fused_plant_comp.GenericHubWindSpeed=fusedwind.plant_flow.fused_plant_comp:GenericHubWindSpeed\nfused_wake.noj.NOJWindFarmWake=fused_wake.noj:NOJWindFarmWake\nfused_wake.wake.PostProcessWindRose=fused_wake.wake:PostProcessWindRose\nfused_wake.wake.HubCenterWS=fused_wake.wake:HubCenterWS\nfused_wake.wake.HomogeneousInflowGenerator=fused_wake.wake:HomogeneousInflowGenerator\nfused_wake.wake.GenericEngineeringWakeModel=fused_wake.wake:GenericEngineeringWakeModel\nfusedwind.plant_flow.fused_plant_comp.GenericInflowGenerator=fusedwind.plant_flow.fused_plant_comp:GenericInflowGenerator\nfused_wake.wake.GenericWindFarm=fused_wake.wake:GenericWindFarm\nfused_wake.noj.MozaicTileWindFarmWake=fused_wake.noj:MozaicTileWindFarmWake\nfused_wake.noj.MozaicTileWakeModel=fused_wake.noj:MozaicTileWakeModel\nfused_wake.wake.GenericInflowGenerator=fused_wake.wake:GenericInflowGenerator\nfused_wake.wake.GenericAEP=fused_wake.wake:GenericAEP\nfusedwind.plant_flow.fused_plant_comp.GenericWakeSum=fusedwind.plant_flow.fused_plant_comp:GenericWakeSum\nfused_wake.wake.AEP=fused_wake.wake:AEP\nfused_wake.wake.LinearWakeSum=fused_wake.wake:LinearWakeSum\nfusedwind.plant_flow.fused_plant_comp.PostProcessWindRose=fusedwind.plant_flow.fused_plant_comp:PostProcessWindRose',
 'include_package_data': True,
 'install_requires': ['openmdao.main'],
 'keywords': ['openmdao'],
 'license': 'Research collaboration',
 'maintainer': 'Pierre-Elouan Rethore, DTU Wind Energy',
 'maintainer_email': 'pire@dtu.dk',
 'name': 'fused_wake',
 'package_data': {'fused_wake': ['sphinx_build/html/.buildinfo',
                                 'sphinx_build/html/genindex.html',
                                 'sphinx_build/html/index.html',
                                 'sphinx_build/html/objects.inv',
                                 'sphinx_build/html/pkgdocs.html',
                                 'sphinx_build/html/py-modindex.html',
                                 'sphinx_build/html/search.html',
                                 'sphinx_build/html/searchindex.js',
                                 'sphinx_build/html/srcdocs.html',
                                 'sphinx_build/html/usage.html',
                                 'sphinx_build/html/_modules/index.html',
                                 'sphinx_build/html/_modules/fused_wake/fused_wake.html',
                                 'sphinx_build/html/_modules/fused_wake/io.html',
                                 'sphinx_build/html/_modules/fused_wake/noj.html',
                                 'sphinx_build/html/_modules/fused_wake/wake.html',
                                 'sphinx_build/html/_modules/fused_wake/windturbine.html',
                                 'sphinx_build/html/_sources/index.txt',
                                 'sphinx_build/html/_sources/pkgdocs.txt',
                                 'sphinx_build/html/_sources/srcdocs.txt',
                                 'sphinx_build/html/_sources/usage.txt',
                                 'sphinx_build/html/_static/ajax-loader.gif',
                                 'sphinx_build/html/_static/basic.css',
                                 'sphinx_build/html/_static/comment-bright.png',
                                 'sphinx_build/html/_static/comment-close.png',
                                 'sphinx_build/html/_static/comment.png',
                                 'sphinx_build/html/_static/default.css',
                                 'sphinx_build/html/_static/doctools.js',
                                 'sphinx_build/html/_static/down-pressed.png',
                                 'sphinx_build/html/_static/down.png',
                                 'sphinx_build/html/_static/file.png',
                                 'sphinx_build/html/_static/jquery.js',
                                 'sphinx_build/html/_static/minus.png',
                                 'sphinx_build/html/_static/plus.png',
                                 'sphinx_build/html/_static/pygments.css',
                                 'sphinx_build/html/_static/searchtools.js',
                                 'sphinx_build/html/_static/sidebar.js',
                                 'sphinx_build/html/_static/underscore.js',
                                 'sphinx_build/html/_static/up-pressed.png',
                                 'sphinx_build/html/_static/up.png',
                                 'sphinx_build/html/_static/websupport.js',
                                 'test/HR_MozaicTileWindFarmWake.pdf',
                                 'test/HR_NOJWindFarmWake.pdf',
                                 'test/noj_single_wake.pdf',
                                 'test/openmdao_log.txt',
                                 'test/test_fused_wake.py',
                                 'test/test_lib.py',
                                 'test/test_NOJ.py',
                                 'test/unittest_ref_files/HR_coordinates.dat',
                                 'test/unittest_ref_files/NOJ/HR_CT_270.txt',
                                 'test/unittest_ref_files/NOJ/HR_Power_270.txt',
                                 'test/unittest_ref_files/NOJ/HR_U_270.txt',
                                 'test/unittest_ref_files/NOJ/points1.txt',
                                 'test/unittest_ref_files/NOJ/points2.txt',
                                 'test/unittest_ref_files/NOJ/points_u1.txt',
                                 'test/unittest_ref_files/NOJ/points_u2.txt']},
 'package_dir': {'': 'src'},
 'packages': ['fused_wake'],
 'url': 'https://github.com/FUSED-Wind/FUSED-Wind/tree/develop/fusedwind_plugins/fused_wake',
 'version': '0.1.1',
 'zip_safe': False}


setup(**kwargs)
