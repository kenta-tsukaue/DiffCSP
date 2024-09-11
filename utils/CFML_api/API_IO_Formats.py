# **************************************************************************
#
# CrysFML API IO Format
#
# @file      Src/API_IO_Formats.py
# @brief     IO Formats based on CrysFML
#
# @homepage  https://code.ill.fr/scientific-software/crysfml
# @license   GNU LGPL (see LICENSE)
# @copyright Institut Laue Langevin 2020-now
# @authors   Scientific Computing Group at ILL (see AUTHORS)
#
# **************************************************************************

import os

import CFML_api.crysfml_api
import CFML_api.API_Crystal_Metrics
import CFML_api.API_Crystallographic_Symmetry
import CFML_api.API_Atom_TypeDef
import CFML_api.FortranBindedClass

class JobInfo(CFML_api.FortranBindedClass):
    """ A class used to handle the job informations, read from a CFL File

    ...
    Attributes
    ----------
    range_2theta : (float, float)
       (theta_min, theta_max) 
    theta_step : float
        delta_theta between two points of the Simulated Powder pattern
    u_resolution : float
       Resolution parameter U for Simulated Powder pattern
    v_resolution : float
       Resolution parameter V for Simulated Powder pattern
    w_resolution : float
       Resolution parameter W for Simulated Powder pattern
    x_resolution : float
       Resolution parameter X for Simulated Powder pattern
    y_resolution : float
       Resolution parameter Y for Simulated Powder pattern
    bkg : float
       Constant background value for Simulated Powder pattern
    
    Methods
    -------
    print_description
        Prints the description of the job

    """
    def __init__(self, string_array=None):
        """
        Initialise a JobInfo. Can take as input an array of strings 

        dat= ['Title SrTiO3',
        'Npatt 1',
        'Patt_1 NEUT_2THE  1.54056    1.54056    1.00      0.0        135.0',
        'UVWXY        0.025  -0.00020   0.01200   0.00150  0.00465',
        'STEP         0.05 ',
        'Backgd       50.000']
        """
        CFML_api.FortranBindedClass.__init__(self)
        if string_array is not None:
            dict = CFML_api.crysfml_api.IO_Formats_jobinfo_from_CIF_string_array(string_array)
            self._set_fortran_address(dict["JobInfo"])
            

    def __del__(self):
        address = self.get_fortran_address()
        if address:
            CFML_api.crysfml_api.IO_Formats_del_jobinfo(self.get_fortran_address())

    def print_description(self):
        """ Prints the job info description """
        
        print(self.title)
        print("Number of patterns: " + str(self.num_patterns) )
        print("Type of pattern: " + self.pattern_type)

        print("Number of phases: " + str(self.num_phases) )
        print("Name of phases: " + str(self.get_phase_names(0)) )
        print("Name of phases: " + str(self.phase_name) )
        print("Lambda range: " +str(self.lambdas) )
        print("Lambda ratio: "+ str(self.lambda_ratio) )
        print("Range 2theta: "+ str(self.range_2theta) )
        print("Range sin(theta)/lambda: "+str(self.range_stl) )
        
        
        
    @property
    def title(self):
        """
        Identification of the job
        """
        return CFML_api.crysfml_api.IO_Formats_get_title(self.get_fortran_address())["title"]
    
    @property
    def num_phases(self):
        """
        Number of phases
        """
        return CFML_api.crysfml_api.IO_Formats_get_num_phases(self.get_fortran_address())["num_phases"]
    
    @property
    def num_patterns(self):
        """
        Number of patterns
        """
        return CFML_api.crysfml_api.IO_Formats_get_num_patterns(self.get_fortran_address())["num_patterns"]
    
    @property
    def num_cmd(self):
        """
        Number of command lines
        """
        return CFML_api.crysfml_api.IO_Formats_get_num_cmd(self.get_fortran_address())["num_cmd"]
    

    def get_pattern_types(self, indx=None):
        """
        String with the pattern type of indx in [0, num_patterns-1]
        """
        if indx:
            key=indx
        else:
            key=0
        return CFML_api.crysfml_api.IO_Formats_get_patt_typ(self.get_fortran_address(), key+1)["patt_typ"]

    def set_pattern_types(self, patt, indx=None):
        """
        String with the pattern type of indx in [0, num_patterns-1]
        """
        if indx:
            key=indx
        else:
            key=0
        return CFML_api.crysfml_api.IO_Formats_set_patt_typ(self.get_fortran_address(), patt, key+1)
    
    pattern_type = property(get_pattern_types, set_pattern_types)

    def get_phase_names(self, indx=None):
        """
        String with the name of the phase indx in [0, num_phases-1]
        """
        if indx:
            key=indx
        else:
            key=0
        return CFML_api.crysfml_api.IO_Formats_get_phas_nam(self.get_fortran_address(), key+1)["phas_nam"]

    phase_name = property(get_phase_names)
    
    
    def get_cmd(self, indx=None):
        """
        Command lines: text for actions in [0, num_cmd-1]
        """
        if indx:
            key=indx
        else:
            key=0
        return CFML_api.crysfml_api.IO_Formats_get_cmd(self.get_fortran_address(),key+1)["cmd"]
    

    def get_range_stl(self,indx=None):
        """
        Range in sinTheta/Lambda of phase indx [0, num_phases-1]
        """
        if indx:
            key=indx
        else:
            key=0
        mina = CFML_api.crysfml_api.IO_Formats_get_range_stl(self.get_fortran_address(),key+1)["min"]
        maxb = CFML_api.crysfml_api.IO_Formats_get_range_stl(self.get_fortran_address(),key+1)["max"]
            
        return mina, maxb

    range_stl = property(get_range_stl)
    

    def get_range_q(self,indx=None):
        """
        Range in 4pi*sinTheta/Lambda of phase indx [0, num_phases-1]
        """
        if indx:
            key=indx
        else:
            key=0
        mina = CFML_api.crysfml_api.IO_Formats_get_range_q(self.get_fortran_address(),key+1)["min"]
        maxb = CFML_api.crysfml_api.IO_Formats_get_range_q(self.get_fortran_address(),key+1)["max"]
            
        return mina, maxb

    range_q = property(get_range_q)
    

    def get_range_d(self, indx=None):
        """
        Range in d-spacing of phase indx [0, num_phases-1]
        """
        if indx:
            key=indx
        else:
            key=0
        mina = CFML_api.crysfml_api.IO_Formats_get_range_d(self.get_fortran_address(),key+1)["min"]
        maxb = CFML_api.crysfml_api.IO_Formats_get_range_d(self.get_fortran_address(),key+1)["max"]
            
        return mina, maxb

    range_d = property(get_range_d)
    
    
    def get_range_2theta(self, indx=None):
        """
        Range in 2theta-spacing of phase indx [0, num_phases-1]
        """
        if indx:
            key=indx
        else:
            key=0
        mina = CFML_api.crysfml_api.IO_Formats_get_range_2theta(self.get_fortran_address(),key+1)["min"]
        maxb = CFML_api.crysfml_api.IO_Formats_get_range_2theta(self.get_fortran_address(),key+1)["max"]
            
        return mina, maxb
        
    
    def set_range_2theta(self, range_theta, indx=None):
        if indx:
            key=indx
        else:
            key=0

        (mina, maxb) = range_theta
        CFML_api.crysfml_api.IO_Formats_set_range_2theta(self.get_fortran_address(), mina, maxb, key+1)

    range_2theta = property(get_range_2theta, set_range_2theta)
    
    
    def get_range_energy(self, indx=None):
        """
        Range in Energy of phase indx [0, num_phases-1]
        """
        if indx:
            key=indx
        else:
            key=0
        mina = CFML_api.crysfml_api.IO_Formats_get_range_energy(self.get_fortran_address(),key+1)["min"]
        maxb = CFML_api.crysfml_api.IO_Formats_get_range_energy(self.get_fortran_address(),key+1)["max"]
            
        return mina, maxb
        
    range_energy = property(get_range_energy)
    
    
    def get_lambdas(self, indx=None):
        """
        Lambda1, Lambda2 of phase indx [0, num_phases-1]
        """
        if indx:
            key=indx
        else:
            key=0
        lambda1 = CFML_api.crysfml_api.IO_Formats_get_lambda(self.get_fortran_address(),key+1)["lambda1"]
        lambda2 = CFML_api.crysfml_api.IO_Formats_get_lambda(self.get_fortran_address(),key+1)["lambda2"]
            
        return lambda1, lambda2

    def set_lambdas(self, range_lambda, indx=None):
        if indx:
            key=indx
        else:
            key=0

        (mina, maxb) = range_lambda
        CFML_api.crysfml_api.IO_Formats_set_lambda(self.get_fortran_address(), mina, maxb, key+1)
    
    lambdas = property(get_lambdas, set_lambdas)
        

    def get_lambda_ratio(self, indx=None):
        """
        ratio lambda2/lambda1  [0, num_phases-1]
        """
        if indx:
            key=indx
        else:
            key=0
        return CFML_api.crysfml_api.IO_Formats_get_ratio(self.get_fortran_address(),key+1)["ratio"]

    def set_lambda_ratio(self, ratio, indx=None):
        if indx:
            key=indx
        else:
            key=0
        CFML_api.crysfml_api.IO_Formats_set_ratio(self.get_fortran_address(), ratio, key+1)
    
    lambda_ratio = property(get_lambda_ratio, set_lambda_ratio)
    
    @property
    def d_to_tof_1(self):
        """
        d-to-TOF coefficient
        """
        return CFML_api.crysfml_api.IO_Formats_get_dtt1(self.get_fortran_address())["dtt1"]
    
    @property
    def d_to_tof_2(self):
        """
        d-to-TOF coefficient
        """
        return CFML_api.crysfml_api.IO_Formats_get_dtt2(self.get_fortran_address())["dtt2"]

    
    @property
    def u_resolution(self):
        """
        Resolution parameter U for Simulated Powder pattern
        """
        return CFML_api.crysfml_api.IO_Formats_get_U(self.get_fortran_address())["U"]
    
    @u_resolution.setter
    def u_resolution(self, val):

        CFML_api.crysfml_api.IO_Formats_set_U(self.get_fortran_address(), val)
    
    @property
    def v_resolution(self):
        """
        Resolution parameter V for Simulated Powder pattern
        """
        return CFML_api.crysfml_api.IO_Formats_get_V(self.get_fortran_address())["V"]

    @v_resolution.setter
    def v_resolution(self, val):

        CFML_api.crysfml_api.IO_Formats_set_V(self.get_fortran_address(), val)
    
    @property
    def w_resolution(self):
        """
        Resolution parameter W for Simulated Powder pattern
        """
        return CFML_api.crysfml_api.IO_Formats_get_W(self.get_fortran_address())["W"]

    @w_resolution.setter
    def w_resolution(self, val):

        CFML_api.crysfml_api.IO_Formats_set_W(self.get_fortran_address(), val)
    
    @property
    def x_resolution(self):
        """
        Resolution parameter X for Simulated Powder pattern
        """
        return CFML_api.crysfml_api.IO_Formats_get_X(self.get_fortran_address())["X"]

    @x_resolution.setter
    def x_resolution(self, val):

        CFML_api.crysfml_api.IO_Formats_set_X(self.get_fortran_address(), val)
    
    @property
    def y_resolution(self):
        """
        Resolution parameter Y for Simluated Powder pattern
        """
        return CFML_api.crysfml_api.IO_Formats_get_Y(self.get_fortran_address())["Y"]

    @y_resolution.setter
    def y_resolution(self, val):

        CFML_api.crysfml_api.IO_Formats_set_Y(self.get_fortran_address(), val)
    
    @property
    def theta_step(self):
        """
        Step in theta for the Simulated Powder patttern
        """
        return CFML_api.crysfml_api.IO_Formats_get_theta_step(self.get_fortran_address())["theta_step"]

    @theta_step.setter
    def theta_step(self, val):

        CFML_api.crysfml_api.IO_Formats_set_theta_step(self.get_fortran_address(), val)
    
    @property
    def bkg(self):
        """
        Cosntant background value for the Simulated Powder patttern
        """
        return CFML_api.crysfml_api.IO_Formats_get_bkg(self.get_fortran_address())["bkg"]

    @bkg.setter
    def bkg(self, val):

        CFML_api.crysfml_api.IO_Formats_set_bkg(self.get_fortran_address(), val)
    

class CIFFile():
    """ A class used to handle CIF Files

    ...
    Attributes
    ----------
    filename : string

    Methods
    -------
    get_as_string_array

    """
    def __init__(self, filename):
        """
        Parameters
        ----------
        filename : str
           Name of the CIF File
        """
        self.__load_cif_file(filename)
    
    @property
    def filename(self):
        """
        Name of the CIF File (str)
        """
        return self.__filename
    
    @property
    def cell(self):
        """
        Cell (CFML_api.Cell type)
        """
        return self.__cell
    
    @property
    def space_group(self):
        """
        Space Group (CFML_api.SpaceGroup type)
        """
        return self.__space_group
    
    @property
    def atom_list(self):
        """
        List of atoms (CFML_api.AtomList type)
        """
        return self.__atom_list

    @property
    def job_info(self):
        """
        Info on the type of job (CFML_api.Job_info type)
        """
        return self.__job_info

    @filename.setter
    def filename(self, filename):
        """
        Setter for the name of the cif file
        The setter reloads the cell, space group and atom list
        """
        self.__load_cif_file(filename)    

    def get_as_string_array(self):
        """
        Returns the CIF File as an array of string
        """
        ret = []
        with open(self.__filename, 'r') as file:
            for line in file.readlines():
                ret.append(line[:-1])
        return ret

    def __load_cif_file(self, filename):
        """
        Internal method
        Loads the CIF File
        """
        self.__filename = filename

        # Check if file exists
        file_exists = False
        if filename[-4:] == ".cif" or filename[-4:] == ".cfl":
            if os.path.exists(filename):
                file_exists = True
                if filename[-4:] == ".cif":
                    mode = "CIF"
                else:
                    mode = "CFL"
        elif os.path.exists(filename+".cfl"):
                filename = filename + ".cfl"
                file_exists = True
                mode = "CFL"
        elif os.path.exists(filename+".cif"):
                filename = filename + ".cif"
                file_exists = True
                mode = "CIF"

        if not file_exists:
            raise IOError("No file with name: " + filename)
        else:
                
            dict = CFML_api.crysfml_api.IO_Formats_readn_set_xtal_structure(self.__filename, mode)
            
            self.__cell = CFML_api.API_Crystal_Metrics.Cell.from_fortran_address(dict["Cell"])
            self.__space_group = CFML_api.API_Crystallographic_Symmetry.SpaceGroup.from_fortran_address(dict["SpG"])
            self.__atom_list = CFML_api.API_Atom_TypeDef.AtomList.from_fortran_address(dict["A"])

            self.__job_info = JobInfo.from_fortran_address(dict["JobInfo"])

