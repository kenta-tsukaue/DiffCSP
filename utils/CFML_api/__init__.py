# **************************************************************************
#
# CrysFML API
#
# @file      Src/__init__.py
# @brief     __init__ for API module
#
# @homepage  https://code.ill.fr/scientific-software/crysfml
# @license   GNU LGPL (see LICENSE)
# @copyright Institut Laue Langevin 2020-now
# @authors   Scientific Computing Group at ILL (see AUTHORS)
#
# **************************************************************************

# Try to import fortran binding
try:
    import CFML_api.crysfml_api
except ImportError as e:
    raise ImportError(str(e) + "\n\n=> Fortran binding could not be found. It may not be properly compiled, or it may be linked with another Python interpreter")

from CFML_api.FortranBindedClass import FortranBindedClass
from CFML_api.API_Atom_TypeDef import AtomList
from CFML_api.API_Atom_TypeDef import Atom
from CFML_api.API_Crystal_Metrics import Cell, k_to_cart_vector, k_to_cart_unitary_vector, betas_from_biso
from CFML_api.API_Crystallographic_Symmetry import SpaceGroup, multiplicity_pos, occupancy_site
from CFML_api.API_Diffraction_Patterns import DiffractionPattern
from CFML_api.API_IO_Formats import CIFFile, JobInfo
from CFML_api.API_Reflections_Utilities import ReflectionList, Reflection
from CFML_api.PowderPatternSimulation import PowderPatternSimulationConditions
from CFML_api.PowderPatternSimulation import PowderPatternSimulationSource
from CFML_api.API_Error_Messages import ErrorMessages

API_version = 0.2
