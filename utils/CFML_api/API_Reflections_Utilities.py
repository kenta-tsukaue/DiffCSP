# **************************************************************************
#
# CrysFML API Reflection Utilities
#
# @file      Src/API_Reflections_Utilities.py
# @brief     Reflections utilities based on CrysFML
#
# @homepage  https://code.ill.fr/scientific-software/crysfml
# @license   GNU LGPL (see LICENSE)
# @copyright Institut Laue Langevin 2020-now
# @authors   Scientific Computing Group at ILL (see AUTHORS)
#
# **************************************************************************

import CFML_api.crysfml_api
import CFML_api.FortranBindedClass

class ReflectionList(CFML_api.FortranBindedClass):
    """ 
    A class used to describe the reflection list type(Reflection_List_Type) in CFML.

    ... 
    Attributes
    ----------
    cell : CMFL_api.crysfml_api.Cell
    spg : CMFL_api.crysfml_api.SpaceGroup
    lfriedel : boolean
    job : CMFL_api.crysfml_api.JobInfo

    Methods
    -------
    print_description
        Prints the reflection lsit
    compute_structure_factors
        Compute the structure factor for each reflection of the list
    
    """
    
    def __init__(self, cell, spg, lfriedel, job):
        CFML_api.FortranBindedClass.__init__(self)
        (stlmin, stlmax) = job.range_stl
         
        self._set_fortran_address(
            CFML_api.crysfml_api.reflections_utilities_hkl_uni_reflist(
                cell.get_fortran_address(), spg.get_fortran_address(),
                lfriedel, stlmin, stlmax)["address"])
        
    def __del__(self):
        CFML_api.crysfml_api.reflections_utilities_del_reflection_list(self.get_fortran_address())

    def __getitem__(self,key):
        dict = CFML_api.crysfml_api.reflections_utilities_get_item(self.get_fortran_address(),key)
        ref = CFML_api.API_Reflections_Utilities.Reflection.from_fortran_address(dict["Ref"])
        return ref

    @property
    def nref(self):
        return CFML_api.crysfml_api.reflections_utilities_get_nref(self.get_fortran_address())["nref"]
    
    def compute_structure_factors(self, space_group, atom_list, job):
        """
        Compute all structure factors 
        ...
        Parameters
        ----------
        space_group : CMFL_api.crysfml_api.SpaceGroup
        atom_list :  CMFL_api.crysfml_api.AtomList
        job : CMFL_api.crysfml_api.JobInfo
        """
        CFML_api.crysfml_api.structure_factors_structure_factors(
            atom_list.get_fortran_address(), space_group.get_fortran_address(),
            self.get_fortran_address(), job.get_fortran_address())
        
    def compute_af0(self, space_group, atom_list, job):
        """
        Compute all structure factors 
        ...
        Parameters
        ----------
        space_group : CMFL_api.crysfml_api.SpaceGroup
        atom_list :  CMFL_api.crysfml_api.AtomList
        job : CMFL_api.crysfml_api.JobInfo
        """
        a = CFML_api.crysfml_api.create_table_af0_xray_fun(
            atom_list.get_fortran_address(), space_group.get_fortran_address(),
            self.get_fortran_address(), job.get_fortran_address())
        return a

        
    def print_description(self):
        """
        Write the list of reflections and structure factors
        """
        print("\n")
        CFML_api.crysfml_api.reflections_utilities_write_reflist_info(self.get_fortran_address())
        
        
class Reflection(CFML_api.FortranBindedClass):
    """ Class for a given Reflection in CFML

    ... 
    Attributes
    ----------

    Methods
    -------
    """
    
    def __init__(self):
        CFML_api.FortranBindedClass.__init__(self)

    def __del__(self):
        CFML_api.crysfml_api.reflections_utilities_del_reflection(self.get_fortran_address())
    
    @property
    def hkl(self):
        """
        hkl indices
        """
        return CFML_api.crysfml_api.reflections_utilities_get_H(self.get_fortran_address())["H"]
    
    @property
    def multiplicity(self):
        """
        Multiplicity
        """
        return CFML_api.crysfml_api.reflections_utilities_get_Mult(self.get_fortran_address())["Mult"]
    
    @property
    def fobs(self):
        """
        Observed structure factor
        """
        return CFML_api.crysfml_api.reflections_utilities_get_Fo(self.get_fortran_address())["Fo"]
    
    @property
    def fcalc(self):
        """
        Calculated structure factor
        """
        return CFML_api.crysfml_api.reflections_utilities_get_Fc(self.get_fortran_address())["Fc"]
    
    @property
    def sigma_fobs(self):
        """
        Sigma for the Observed structure factor
        """
        return CFML_api.crysfml_api.reflections_utilities_get_SFo(self.get_fortran_address())["SFo"]
    
    @property
    def stl(self):
        """
        sin(theta)/lambda
        """
        return CFML_api.crysfml_api.reflections_utilities_get_S(self.get_fortran_address())["S"]
    
    @property
    def weight(self):
        """
        weight
        """
        return CFML_api.crysfml_api.reflections_utilities_get_W(self.get_fortran_address())["W"]
    
    @property
    def phase(self):
        """
        Phase in degrees
        """
        return CFML_api.crysfml_api.reflections_utilities_get_Phase(self.get_fortran_address())["Phase"]
    
    @property
    def a(self):
        """
        real part of the Structure Factor
        """
        return CFML_api.crysfml_api.reflections_utilities_get_A(self.get_fortran_address())["A"]
    
    @property
    def b(self):
        """
        imaginary part of the structure factor
        """
        return CFML_api.crysfml_api.reflections_utilities_get_B(self.get_fortran_address())["B"]
    
