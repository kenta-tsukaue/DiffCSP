# **************************************************************************
#
# CrysFML API Crystallographic Symmetry
#
# @file      Src/API_Crystallographic_Symmetry.py
# @brief     Symmetry groups properties based on CrysFML
#
# @homepage  https://code.ill.fr/scientific-software/crysfml
# @license   GNU LGPL (see LICENSE)
# @copyright Institut Laue Langevin 2020-now
# @authors   Scientific Computing Group at ILL (see AUTHORS)
#
# **************************************************************************

import CFML_api.crysfml_api
import CFML_api.FortranBindedClass

class CentricType():
    CENTRIC = 0
    ACENTRIC = 1
    CENTRIC_EDGE_AT_ORIGIN = 2

class SpaceGroup(CFML_api.FortranBindedClass):
    """ A class used to describe a Crystallographic Space Group type(Space_group_type) in CFML

    ...
    Attributes
    ----------
    group_id : int
       Space group number listed in the International Tables for Crystallography

    Methods
    -------
    print_description
        Prints the lattice cell description

    """
    def __init__(self, group_id=None):
        """
        Parameters
        ----------
        group_id : int
            Space group number
        """
        CFML_api.FortranBindedClass.__init__(self)
        if group_id is not None:
            self._set_fortran_address(CFML_api.crysfml_api.crystallographic_symmetry_set_spacegroup(group_id)["address"])

    def __del__(self):
        address = self.get_fortran_address()
        if address:
            CFML_api.crysfml_api.crystallographic_symmetry_del_spacegroup(address)

    def print_description(self):
        """ Prints the lattice cell description """
        CFML_api.crysfml_api.crystallographic_symmetry_write_spacegroup(self.get_fortran_address())

    @property
    def lattice_translation(self):
        """
        Lattice translation matrix
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_latt_trans(self.get_fortran_address())["latt_trans"]
    
    @property
    def is_hexa(self):
        """
        Is space group hexagonal
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_hexa(self.get_fortran_address())["hexa"]
    
    @property
    def space_group_number(self):
        """
        Number of the space group
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_numspg(self.get_fortran_address())["numspg"]
    
    @property
    def space_group_symbol(self):
        """
        Hermann-Mauguin Symbol (str)
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_spg_symb(self.get_fortran_address())["spg_symb"]
    
    @property
    def hall_symbol(self):
        """
        Hall symbol
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_hall(self.get_fortran_address())["hall"]
    
    @property
    def generalized_hall_symbol(self):
        """
        Generalised Hall symbol
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_ghall(self.get_fortran_address())["ghall"]
    
    @property
    def crystal_system(self):
        """
        Crystal System
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_CrystalSys(self.get_fortran_address())["CrystalSys"]
    
    @property
    def laue_class(self):
        """
        Laue class
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_Laue(self.get_fortran_address())["Laue"]
    
    @property
    def point_group(self):
        """
        Point group
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_PG(self.get_fortran_address())["PG"]
    
    @property
    def extra_information(self):
        """
        Extra Information
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_Info(self.get_fortran_address())["Info"]
    
    @property
    def space_group_setting(self):
        """
        Information about the SG setting (IT,KO,ML,ZA,Table,Standard,UnConventional)
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_SG_setting(self.get_fortran_address())["SG_setting"]
    
    @property
    def space_group_lattice_type(self):
        """
        Lattice type
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_SPG_lat(self.get_fortran_address())["SPG_lat"]
    
    @property
    def space_group_lattice_type_symbol(self):
        """
        Lattice type Symbol
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_SPG_latsy(self.get_fortran_address())["SPG_latsy"]
    
    @property
    def number_of_lattice_points(self):
        """
        Number of lattice points in a cell
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_NumLat(self.get_fortran_address())["NumLat"]
    
    @property
    def bravais(self):
        """
        String with Bravais symbol + translations
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_bravais(self.get_fortran_address())["bravais"]
    
    @property
    def centre(self):
        """
        Alphanumeric information about the center of symmetry
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_centre(self.get_fortran_address())["centre"]
    
    @property
    def centric(self):
        """
        Centric or Acentric [ =0 Centric(-1 no at origin),=1 Acentric,=2 Centric(-1 at origin)]
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_centred(self.get_fortran_address())["centred"]
    
    @property
    def inversion_centre(self):
        """
        Fractional coordinates of the inversion centre
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_centre_coord(self.get_fortran_address())["centre_coord"]
    
    @property
    def number_of_symops(self):
        """
        Number of reduced set of Symmetry Operators
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_numops(self.get_fortran_address())["numops"]
    
    @property
    def multiplicity(self):
        """
        Multiplicity of the general position
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_multip(self.get_fortran_address())["multip"]
    
    @property
    def operators_minimum_number(self):
        """
        Minimum number of operators to generate the Group
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_num_gen(self.get_fortran_address())["num_gen"]
    
    @property
    def symmetry_operators(self):
        """
        Symmetry operators (192)
        """
        symmetry_operators = []
        for k in range(self.multiplicity):
            symmetry_operator = SymmetryOperator.from_fortran_address(CFML_api.crysfml_api.crystallographic_symmetry_get_SymOp(self.get_fortran_address(), k+1)["SymOp"])
            symmetry_operator.set_parent(self)
            symmetry_operators.append(symmetry_operator)
            #symmetry_operators.append(SymmetryOperator(CFML_api.crysfml_api.crystallographic_symmetry_get_SymOp(self.get_fortran_address(), k+1)["SymOp"], parent=self))
        return symmetry_operators
    
    @property
    def symmetry_operators_as_text(self):
        """
        String form of symmetry operators
        """
        symmetry_operators_as_text = []
        for k in range(self.multiplicity):
            symmetry_operators_as_text.append(CFML_api.crysfml_api.crystallographic_symmetry_get_SymopSymb(self.get_fortran_address(), k+1)["SymopSymb"])
        return symmetry_operators_as_text
    
    @property
    def wyckoff_info(self):
        """
        Wyckoff Information
        """
        wyckoff_type = WyckoffType.from_fortran_address(CFML_api.crysfml_api.crystallographic_symmetry_get_wyckoff(self.get_fortran_address())["wyckoff"])
        wyckoff_type.set_parent(self) 
        return wyckoff_type
    
    @property
    def asymetric_unit(self):
        """
        Asymmetric unit in real space
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_R_asym_unit(self.get_fortran_address())["R_asym_unit"]
    
class SymmetryOperator(CFML_api.FortranBindedClass):    
    @property
    def rotation_matrix(self):
        """
        Rotation matrix of the operator
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_symmetry_operator_rotation_matrix(self.get_fortran_address())["rotation_matrix"]
        
    @property
    def translation_matrix(self):
        """
        Translation matrix of the operator
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_symmetry_operator_trans_matrix(self.get_fortran_address())["translation_matrix"]
        
        
class WyckoffType(CFML_api.FortranBindedClass):
    @property
    def num_orbit(self):
        #t = CFML_api.crysfml_api.crystallographic_symmetry_get_wyckoff_num_orbit(self.get_fortran_address())
        #print(t)
        return CFML_api.crysfml_api.crystallographic_symmetry_get_wyckoff_num_orbit(self.get_fortran_address())["num_orbit"]
    
    @property
    def orbits(self):
        orbits = []
        for k in range(26):
            orbit = WyckoffOrbit.from_fortran_address(CFML_api.crysfml_api.crystallographic_symmetry_get_wyckoff_orbits(self.get_fortran_address(), k+1)["orbit"])
            orbit.set_parent(self)
            orbits.append(orbit)
        return orbits
    
class WyckoffOrbit(CFML_api.FortranBindedClass):
    @property
    def multiplicity(self):
        """
        Multiplicity of the orbit
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_wyckoff_multp(self.get_fortran_address())["multp"]
    
    @property
    def site(self):
        """
        Site (str)
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_wyckoff_site(self.get_fortran_address())["site"]
    
    @property
    def orbit_number(self):
        """
        Number of the orbit
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_wyckoff_norb(self.get_fortran_address())["norb"]
    
    @property
    def str_orig(self):
        """
        Str_orig (str)
        """
        return CFML_api.crysfml_api.crystallographic_symmetry_get_wyckoff_str_orig(self.get_fortran_address())["str_orig"]
    
    @property
    def str_orbit(self):
        ret = []
        for k in range(48):
            ret.append(CFML_api.crysfml_api.crystallographic_symmetry_get_wyckoff_str_orbit(self.get_fortran_address(), k+1)["str_orbit"])
        return ret



def multiplicity_pos(pos, spg: SpaceGroup):
    """
    Return the multiplicity of a real space point given the Shubnikov/space group.
    ...
    Parameters
    ----------
    pos : np.array[dtype='float32', ndim=q]
       x,y,z coordinates 
    spg : CMFL_api.crysfml_api.SpaceGroup

    Returns:
    int: Multiplicity of the point

    """
    return CFML_api.crysfml_api.crystallographic_symmetry_get_multip_pos_crys(pos, spg.get_fortran_address())["mult"]


def occupancy_site(pos, spg):
    """
    Return the occupancy foctor given the position and Shubnikov/space group.
    ...
    Parameters
    ----------
    pos : np.array[dtype='float32', ndim=1]
       x,y,z coordinates 
    spg : CMFL_api.crysfml_api.SpaceGroup

    Returns:
    float: occupancy factor (site multiplicity/multiplicity)

    """
    return CFML_api.crysfml_api.crystallographic_symmetry_get_occ_site(pos, spg.get_fortran_address())["occ"]
