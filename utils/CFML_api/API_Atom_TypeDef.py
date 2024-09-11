# **************************************************************************
#
# CrysFML API Atoms
#
# @file      Src/API_Atom_TypeDef.py
# @brief     Atoms properties based on CrysFML
#
# @homepage  https://code.ill.fr/scientific-software/crysfml
# @license   GNU LGPL (see LICENSE)
# @copyright Institut Laue Langevin 2020-now
# @authors   Scientific Computing Group at ILL (see AUTHORS)
#
# **************************************************************************

import CFML_api.crysfml_api
import CFML_api.FortranBindedClass

class Atom(CFML_api.FortranBindedClass):
    """ Class for a given aton type(Atom_type) in CFML

    ... 
    Attributes
    ----------

    Methods
    -------
    """
    
    def __init__(self, string=None):
        CFML_api.FortranBindedClass.__init__(self)
        if string:
            self.from_string(string)
        
    def __del__(self):
        address = self.get_fortran_address()
        if address:
            CFML_api.crysfml_api.atom_typedef_del_atom(address)

    def from_string(self, string):
        dict = CFML_api.crysfml_api.atom_typedef_atom_from_string(string)
        self._set_fortran_address(dict["Atom"])
        
    def __str__(self):
        xyz = self.xyz
        return 'ATOM %s %s %s %s %s %s %s %s %s'%(self.label, self.chemical_symbol, str(xyz[0]), str(xyz[1]), str(xyz[2]), str(self.biso), str(self.occ), str(self.moment), str(self.formal_charge))
    
    @property
    def label(self):
        """
        label
        """
        return CFML_api.crysfml_api.atom_typedef_get_Lab(self.get_fortran_address())["Lab"]
    
    @property
    def chemical_symbol(self):
        """
        Chemical Symbol
        """
        return CFML_api.crysfml_api.atom_typedef_get_ChemSymb(self.get_fortran_address())["ChemSymb"]
    
    @property
    def sfac_symbol(self):
        """
        Symbol for Scattering Factor
        """
        return CFML_api.crysfml_api.atom_typedef_get_SfacSymb(self.get_fortran_address())["SfacSymb"]
    
    @property
    def wyckoff_letter(self):
        """
        Wyckoff letter
        """
        return CFML_api.crysfml_api.atom_typedef_get_wyck(self.get_fortran_address())["wyck"]
    
    def is_active(self):
        """
        Flag to control is the atom is included or excluded
        """
        return CFML_api.crysfml_api.atom_typedef_get_Active(self.get_fortran_address())["Active"]
    
    @property
    def atomic_number(self):
        """
        Atomic number
        """
        return CFML_api.crysfml_api.atom_typedef_get_Z(self.get_fortran_address())["Z"]
    
    @property
    def site_multiplicity(self):
        """
        Multiplicity of the site
        """
        return CFML_api.crysfml_api.atom_typedef_get_Mult(self.get_fortran_address())["Mult"]
    
    @property
    def xyz(self):
        """
        Fractional coordinates
        """
        return CFML_api.crysfml_api.atom_typedef_get_X(self.get_fortran_address())["X"]
    
    @property
    def xyz_std_dev(self):
        """
        Standard deviation of the fractional coordinates
        """
        return CFML_api.crysfml_api.atom_typedef_get_X_Std(self.get_fortran_address())["X_Std"]
    
    @property
    def multiplier_xyz(self):
        """
        Multipliers for xyz
        """
        return CFML_api.crysfml_api.atom_typedef_get_MX(self.get_fortran_address())["MX"]
    
    @property
    def lsq_xyz(self):
        """
        Labels in the LSQ list (refined parameters) for xyz
        """
        return CFML_api.crysfml_api.atom_typedef_get_LX(self.get_fortran_address())["LX"]
    
    @property
    def occ(self):
        """
        Occupation factor
        """
        return CFML_api.crysfml_api.atom_typedef_get_Occ(self.get_fortran_address())["Occ"]
    
    @property
    def occ_std_dev(self):
        """
        Standard deviation of the occupation factor
        """
        return CFML_api.crysfml_api.atom_typedef_get_Occ_Std(self.get_fortran_address())["Occ_Std"]
    
    @property
    def multiplier_occ(self):
        """
        Multiplier for occ
        """
        return CFML_api.crysfml_api.atom_typedef_get_MOcc(self.get_fortran_address())["MOcc"]
    
    @property
    def lsq_occ(self):
        """
        Label in the LSQ list for occ
        """
        return CFML_api.crysfml_api.atom_typedef_get_LOcc(self.get_fortran_address())["LOcc"]
    
    @property
    def biso(self):
        """
        Isotropic B-factor
        """
        return CFML_api.crysfml_api.atom_typedef_get_Biso(self.get_fortran_address())["Biso"]
    
    @property
    def biso_std_dev(self):
        """
        Standard deviation for biso
        """
        return CFML_api.crysfml_api.atom_typedef_get_Biso_std(self.get_fortran_address())["Biso_std"]
    
    @property
    def multiplier_biso(self):
        """
        Multiplier for biso
        """
        return CFML_api.crysfml_api.atom_typedef_get_MBiso(self.get_fortran_address())["MBiso"]
    
    @property
    def lsq_biso(self):
        """
        Label in the LSQ list for biso
        """
        return CFML_api.crysfml_api.atom_typedef_get_LBiso(self.get_fortran_address())["LBiso"]
    
    @property
    def u_type(self):
        """
        Type of anisotropic thermal parameters: u_ij, b_ij, beta, none
        """
        return CFML_api.crysfml_api.atom_typedef_get_Utype(self.get_fortran_address())["Utype"]
    
    @property
    def thermal_type(self):
        """
        Type of behavior wrt to the temperature: isotr, aniso or other
        """
        return CFML_api.crysfml_api.atom_typedef_get_ThType(self.get_fortran_address())["ThType"]
    
    @property
    def u_array(self):
        """
        Array of anisotropic thermal coefficients U11, U22, U33, U12, U13, U23 
        or B11, B22, B33, B12, B13, B23 depending on the value of Utype
        """
        return CFML_api.crysfml_api.atom_typedef_get_U(self.get_fortran_address())["U"]
    
    @property
    def u_array_std_dev(self):
        """
        Array of standard deviation of the anisotropic thermal coefficients
        """
        return CFML_api.crysfml_api.atom_typedef_get_U_std(self.get_fortran_address())["U_std"]
    
    @property
    def multiplier_u_array(self):
        """
        Array of multipliers for the anisotropic thermal coefficients
        """
        return CFML_api.crysfml_api.atom_typedef_get_MU(self.get_fortran_address())["MU"]
    
    @property
    def lsq_u_qrrqy(self):
        """
        Array of labels in the LSQ list for the anisotropic thermal coefficients
        """
        return CFML_api.crysfml_api.atom_typedef_get_LU(self.get_fortran_address())["LU"]
    
    @property
    def u_equiv(self):
        """
        Equivalent isotropic atomic displacement parameter, U(equiv), in angstroms squared,
        calculated from anisotropic atomic displacement parameters. 
        U(equiv) = (1/3) sum~i~[sum~j~(U^ij^ a*~i~ a*~j~ a~i~ a~j~)]
        """
        return CFML_api.crysfml_api.atom_typedef_get_Ueq(self.get_fortran_address())["Ueq"]
    
    @property
    def formal_charge(self):
        """
        The net integer charge assigned to this atom. 
        This is the formal charge assignment normally found in chemical diagrams.
        """
        return CFML_api.crysfml_api.atom_typedef_get_Charge(self.get_fortran_address())["Charge"]
    
    @property
    def moment(self):
        """
        Theoretical value of the magnetic moment. i.e., number of unpaired electrons for 
        transition metal oxides
        """
        return CFML_api.crysfml_api.atom_typedef_get_Moment(self.get_fortran_address())["Moment"]
    
    @property
    def index_sfac(self):
        """
        Index for different purposes 
        (e.g. 1:pointer to the scattering factor, 2:pointer to the magnetic scattering factor
        """
        return CFML_api.crysfml_api.atom_typedef_get_Ind(self.get_fortran_address())["Ind"]
    
    @property
    def nvarF(self):
        """
        Dimension of VarF
        """
        return CFML_api.crysfml_api.atom_typedef_get_NVar(self.get_fortran_address())["NVar"]
    
    @property
    def varF(self):
        """
        Free variables used for different purposes (1,2,3 reserved for occupations, not refinable)
        """
        return CFML_api.crysfml_api.atom_typedef_get_VarF(self.get_fortran_address())["VarF"]
    
    @property
    def multiplier_varF(self):
        """
        Multipliers of the free variables varF
        """
        return CFML_api.crysfml_api.atom_typedef_get_MVarF(self.get_fortran_address())["MVarF"]
    
    @property
    def lsq_varF(self):
        """
        Labels in the LSQ list of the free variables varF
        """
        return CFML_api.crysfml_api.atom_typedef_get_LVarF(self.get_fortran_address())["LVarF"]
    
    @property
    def info(self):
        """
        Information String
        """
        return CFML_api.crysfml_api.atom_typedef_get_AtmInfo(self.get_fortran_address())["AtmInfo"]
    
    @property
    def m_xyz(self):
        """
        Magnetic moment along x,y,z
        """
        return CFML_api.crysfml_api.atom_typedef_get_m_xyz(self.get_fortran_address())["m_xyz"]
    
    @property
    def m_xyz_std_dev(self):
        """
        Standard deviation of the magnetic moment along x,y,z
        """
        return CFML_api.crysfml_api.atom_typedef_get_sm_xyz(self.get_fortran_address())["sm_xyz"]
    
    @property
    def multiplier_m_xyz(self):
        """
        Multipliers of the magnetic moment along x,y,
        """
        return CFML_api.crysfml_api.atom_typedef_get_Mm_xyz(self.get_fortran_address())["Mm_xyz"]
    
    @property
    def lsq_m_xyz(self):
        """
        Label in the LSQ list of the magnetic moment along x,y,
        """
        return CFML_api.crysfml_api.atom_typedef_get_Lm_xyz(self.get_fortran_address())["Lm_xyz"]
    
    @xyz.setter
    def xyz(self, xyz):
        """
        Setter for the position
        Requires a numpy array or a list (length=3) as input
        """
        # Label Element x y z b_iso multiplicity
        string = 'ATOM %s %s %s %s %s %s %s' %(self.label, self.chemical_symbol, str(xyz[0]), str(xyz[1]), str(xyz[2]), self.biso, self.site_multiplicity)
        self.from_string(string)




        
class AtomList(CFML_api.FortranBindedClass):
    """ Class for the list of Atoms type(Atom_list_type) in CFML. 

    ...
    Attributes
    ----------

    Methods
    -------
    print_description
        Print the list of atoms
    """
    def __init__(self, string_array=None):
        CFML_api.FortranBindedClass.__init__(self)
        self._mult = True
        self._adp = True
        if string_array:
            self._mult = False 
            self._adp = False
            
            dict = CFML_api.crysfml_api.atom_typedef_atomlist_from_CIF_string_array(string_array)
            self._set_fortran_address(dict["AtomList"])
              
    def __del__(self):
        CFML_api.crysfml_api.atom_typedef_del_atom_list(self.get_fortran_address())
    
    def __getitem__(self,key):
        dict = CFML_api.crysfml_api.atom_typedef_get_item(self.get_fortran_address(),key)
        atom = CFML_api.API_Atom_TypeDef.Atom.from_fortran_address(dict["Atom"])
        return atom
        
    def __setitem__(self,key,atom):               
        # Destroy old atom
        dict = CFML_api.crysfml_api.atom_typedef_get_item(self.get_fortran_address(),key+1)
        CFML_api.crysfml_api.atom_typedef_del_atom(dict["Atom"])
            
        # Put new atom in Fortran list
        CFML_api.crysfml_api.atom_typedef_set_item(self.get_fortran_address(), atom.get_fortran_address(), key+1)
    
    @property
    def natoms(self):
        """
        Total number of atoms in the list
        """
        return CFML_api.crysfml_api.atom_typedef_get_natoms(self.get_fortran_address())["natoms"]
    
    def print_description(self):
        """ Print the list of atoms. """
        
        CFML_api.crysfml_api.atom_typedef_write_atom_list(self.get_fortran_address())


    def set_mult_occ_cif(self,spg):
        """
        Update the multiplicity and modify the occupancy when reading from a CIF array,
        in order to be in agreement with the definitions of Sfac in CrysFML
        This needs to be run when the AtomList is initialised from a CIF-like stringarray
        rather than from the class CIFFile
        """
        if (self._mult == False):
            CFML_api.crysfml_api.atom_typedef_atomlist_reset_occ_cif(self.get_fortran_address(),spg.get_fortran_address())
            self._mult = True
        else:
            print("Warning: multiplicity and occupancy have already been set")

    def set_all_anisotropic_displacement_parameters_cif(self,cell):
        """
        Update all Atom fields for Us, Bs and betas
        This needs to be run when the AtomList is initialised from a CIF-like stringarray
        rather than from the class CIFFile
        """
        if (self._adp == False):
            CFML_api.crysfml_api.atom_typedef_atomlist_set_all_adp_cif(self.get_fortran_address(),cell.get_fortran_address())
            self._adp = True
        else:
            print("Warning: Anisotropic Displacement Parameters have already been set")
