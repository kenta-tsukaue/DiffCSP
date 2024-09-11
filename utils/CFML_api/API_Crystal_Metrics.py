# **************************************************************************
#
# CrysFML API Crystal
#
# @file      Src/API_Crystal_Metrics.py
# @brief     Crystal properties based on CrysFML
#
# @homepage  https://code.ill.fr/scientific-software/crysfml
# @license   GNU LGPL (see LICENSE)
# @copyright Institut Laue Langevin 2020-now
# @authors   Scientific Computing Group at ILL (see AUTHORS)
#
# **************************************************************************

import CFML_api.crysfml_api
import CFML_api.FortranBindedClass

class Cell(CFML_api.FortranBindedClass):
    """ A class used to describe the crystal cell type(Crystal_Cell_type) in CFML.

    ... 
    Attributes
    ----------
    lattpar : ndarray(dtype='float32', ndim=1)
        Array containing the lattice parameters (a,b,c)
    lattangle : ndarray(dtype='float32', ndim=1)
        Array containing the lattice angles (alpha, beta, gamma)

    Methods
    -------
    print_description
        Prints the lattice cell description
    
    """
    
    def __init__(self, lattpar=None, lattangle=None):
        """
        Parameters
        ----------
        lattpar : ndarray(dtype='float32', ndim=1)
            Array containing the lattice parameters (a,b,c)
        lattangle : ndarray(dtype='float32', ndim=1)
            Array containing the lattice angles (alpha, beta, gamma)
        """
        CFML_api.FortranBindedClass.__init__(self)
        if lattpar is not None and lattangle is not None:
            self._set_fortran_address(
                CFML_api.crysfml_api.crystal_metrics_set_crystal_cell(lattpar,lattangle)["address"])
    
    def __del__(self):
        address = self.get_fortran_address()
        if address:
            CFML_api.crysfml_api.crystal_metrics_del_crystal_cell(address)

    def print_description(self):
        """ Prints the lattice cell description """
        
        CFML_api.crysfml_api.crystal_metrics_write_crystal_cell(self.get_fortran_address())

    @property
    def lattpar(self):
        """ Lattice parameters (a,b,c). (mutable) """
        
        return CFML_api.crysfml_api.crystal_metrics_get_cell(self.get_fortran_address())["cell"]

    @lattpar.setter
    def lattpar(self, lattpar):
        lattangle = CFML_api.crysfml_api.crystal_metrics_get_ang(self.get_fortran_address())["ang"]
        CFML_api.crysfml_api.crystal_metrics_del_crystal_cell(self.get_fortran_address())
        self._set_fortran_address(CFML_api.crysfml_api.crystal_metrics_set_crystal_cell(lattpar,lattangle)["address"])

    @property
    def lattangle(self):
        """ Lattice angles (alpha,beta,gamma). (mutable) """
    
        return CFML_api.crysfml_api.crystal_metrics_get_ang(self.get_fortran_address())["ang"]

    @lattangle.setter
    def lattangle(self, lattangle):
        lattpar = CFML_api.crysfml_api.crystal_metrics_get_cell(self.get_fortran_address())["cell"]
        CFML_api.crysfml_api.crystal_metrics_del_crystal_cell(self.get_fortran_address())
        self._set_fortran_address(CFML_api.crysfml_api.crystal_metrics_set_crystal_cell(lattpar,lattangle)["address"])

    @property
    def lsq_lattpar(self):
        """ Code number for refinement in optimization procedures, label in LSQ list """
    
        return CFML_api.crysfml_api.crystal_metrics_get_lcell(self.get_fortran_address())["lcell"]

    @property
    def lsq_lattangle(self):
        """ Code number for refinement in optimization procedures, label in LSQ list """
    
        return CFML_api.crysfml_api.crystal_metrics_get_lang(self.get_fortran_address())["lang"]
    
    @property
    def lattpar_std_dev(self):
        """ Standard deviations of the cell parameters. """
    
        return CFML_api.crysfml_api.crystal_metrics_get_cell_std(self.get_fortran_address())["cell_std"]
    
    @property
    def lattangle_std_dev(self):
        """ Standard deviations of the cell parameters. """
    
        return CFML_api.crysfml_api.crystal_metrics_get_ang_std(self.get_fortran_address())["ang_std"]
    
    @property
    def reciprocal_lattpar(self):
        """ Reciprocal cell parameters. """
        return CFML_api.crysfml_api.crystal_metrics_get_rcell(self.get_fortran_address())["rcell"]
    
    @property
    def reciprocal_lattangle(self):
        """ Reciprocal cell parameters. """
        return CFML_api.crysfml_api.crystal_metrics_get_rang(self.get_fortran_address())["rang"]
    
    @property
    def direct_metric_tensor(self):
        """ Direct Metric Tensor. """
        return CFML_api.crysfml_api.crystal_metrics_get_GD(self.get_fortran_address())["GD"]
    
    @property
    def reciprocal_metric_tensor(self):
        """ Reciprocal Metric Tensor. """
        return CFML_api.crysfml_api.crystal_metrics_get_GR(self.get_fortran_address())["GR"]
    
    @property
    def crystal_to_orth_matrix(self):
        """ P-Matrix transforming direct Crytal cell to Orthonormal basis """
        return CFML_api.crysfml_api.crystal_metrics_get_Cr_Orth_Cel(self.get_fortran_address())["Cr_Orth_Cel"]
    
    @property
    def orth_to_crystal_matrix(self):
        """ P-Matrix transforming Orthonormal basis to direct Crytal cell """ 
        return CFML_api.crysfml_api.crystal_metrics_get_Orth_Cr_Cel(self.get_fortran_address())["Orth_Cr_Cel"]
    
    @property
    def BL_matrix(self):
        """ Busing-Levy B-matrix 

        transforms hkl to a Cartesian system with x//a*, y in (a*,b*) and z//c 
        """

        return CFML_api.crysfml_api.crystal_metrics_get_BL_M(self.get_fortran_address())["BL_M"]
    
    @property
    def inv_BL_matrix(self):
        """ inverse Busing-Levy B-matrix """
        return CFML_api.crysfml_api.crystal_metrics_get_BL_Minv(self.get_fortran_address())["BL_Minv"]
    
    @property
    def direct_cell_vol(self):
        """ Direct cell volume """
        return CFML_api.crysfml_api.crystal_metrics_get_cellvol(self.get_fortran_address())["cellvol"]
    
    @property
    def reciprocal_cell_vol(self):
        """ Reciprocal cell volume """
        return CFML_api.crysfml_api.crystal_metrics_get_rcellvol(self.get_fortran_address())["rcellvol"]
    
    @property
    def direct_cell_vol_std_dev(self):
        """ Standard deviation of the cell volume """
        return CFML_api.crysfml_api.crystal_metrics_get_stdvol(self.get_fortran_address())["stdvol"]
    
    @property
    def cartesian_frame(self):
        """ Cartesian Frame type: if CartType='A' the Cartesian Frame has x // a. """
        return CFML_api.crysfml_api.crystal_metrics_get_CartType(self.get_fortran_address())["CartType"]


def k_to_cart_vector(code, vector, cell):
    """ 
    Convert a vector in crystal space to cartesian components
    The value of code has been extended to use also the Busing-Levy
    Cartesian system as reference also for direct and reciprocal space.
    Codes:
    The Cartesian frame is that defined by the setting of the Cell object
    D: The components are given with respect to basis (a,b,c)
    R: The components are given with respect to basis (a*,b*,c*)
    BL: The components are given with respect to basis (a*,b*,c*) but
    the Cartesian frame is that defined by Busing and Levy
    BLD: The components are given with respect to basis (a,b,c) but
    the Cartesian frame is that defined by Busing and Levy
    ...
    Parameters
    ----------
    code : str
       Cartesian frame 
    vector : np.array[dtype='float32', ndim=1]
       kx,ky,kz coordinates 
    cell : CMFL_api.crysfml_api.Cell

    Returns:
    np.array[dtype='float32', ndim=1] : coordinates in the chosen cartesian frame

    """
    return CFML_api.crysfml_api.crystal_metrics_cart_vector(code, vector, cell.get_fortran_address())["cart_vec"]

def k_to_cart_unitary_vector(code, vector, cell):
    """
    Convert a vector in crystal space to unitary cartesian components
    The value of code has been extended to use also the Busing-Levy
    Cartesian system as reference also for direct and reciprocal space.
    Codes:
    The Cartesian frame is that defined by the setting of the Cell object
    D: The components are given with respect to basis (a,b,c)
    R: The components are given with respect to basis (a*,b*,c*)
    BL: The components are given with respect to basis (a*,b*,c*) but
    the Cartesian frame is that defined by Busing and Levy
    BLD: The components are given with respect to basis (a,b,c) but
    the Cartesian frame is that defined by Busing and Levy
    ...
    Parameters
    ----------
    code : str
       Cartesian frame 
    vector : np.array[dtype='float32', ndim=1]
       kx,ky,kz coordinates 
    cell : CMFL_api.crysfml_api.Cell

    Returns:
    np.array[dtype='float32', ndim=1] : coordinates in the chosen cartesian frame

    """
    return CFML_api.crysfml_api.crystal_metrics_cart_u_vector(code, vector, cell.get_fortran_address())["cart_vec"]

def betas_from_biso(biso, cell):
    """
    Get Betas from Biso 
    ...
    Parameters
    ----------
    biso : float32
    cell : CMFL_api.crysfml_api.Cell

    Returns
    np.array[dtype='float32', ndim=1] : betas[0:5] b11, b22, b33, b12, b13, b23
    """
    return CFML_api.crysfml_api.crystal_metrics_get_betas_from_biso(cell.get_fortran_address(),biso)["betas"]
