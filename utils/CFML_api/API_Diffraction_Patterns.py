# **************************************************************************
#
# CrysFML API Diffraction Patters
#
# @file      Src/API_Diffraction_Patterns.py
# @brief     Diffraction Patterns utilities based on CrysFML
#
# @homepage  https://code.ill.fr/scientific-software/crysfml
# @license   GNU LGPL (see LICENSE)
# @copyright Institut Laue Langevin 2020-now
# @authors   Scientific Computing Group at ILL (see AUTHORS)
#
# **************************************************************************

import CFML_api.crysfml_api
import CFML_api.FortranBindedClass

class DiffractionPattern(CFML_api.FortranBindedClass):
    def __init__(self, job_info, reflection_list, scale_f):
        CFML_api.FortranBindedClass.__init__(self)
        
        self._set_fortran_address(CFML_api.crysfml_api.diffraction_patterns_compute_powder_pattern(
            job_info.get_fortran_address(), 
            reflection_list.get_fortran_address(), scale_f)["address"])
        
    def __del__(self):
        address = self.get_fortran_address()
        if address:
            CFML_api.crysfml_api.diffraction_patterns_del_powder_pattern(self.get_fortran_address())
    
    @property
    def title(self):
        """
        Identification of the pattern
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_title(self.get_fortran_address())["title"]
    
    @property
    def diff_kind(self):
        """
        type of radiation
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_diff_kind(self.get_fortran_address())["diff_kind"]
    
    @property
    def scat_var(self):
        """
        x-space: 2theta, TOF, Q, s, d-spacing, SinT/L, etc
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_scat_var(self.get_fortran_address())["scat_var"]
    
    @property
    def xax_text(self):
        """
        x-axis legend, eg. Lambda (Angstroms)
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_xax_text(self.get_fortran_address())["xax_text"]
    
    @property
    def yax_text(self):
        """
        y-axis legend, eg. Intensity (arb. units)
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_yax_text(self.get_fortran_address())["yax_text"]
    
    @property
    def instr(self):
        """
        file type
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_instr(self.get_fortran_address())["instr"]
    
    @property
    def filename(self):
        """
        file name
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_filename(self.get_fortran_address())["filename"]
    
    @property
    def filepath(self):
        """
        file path
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_filepath(self.get_fortran_address())["filepath"]
    
    @property
    def xmin(self):
        """
        
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_xmin(self.get_fortran_address())["xmin"]
    
    @property
    def xmax(self):
        """
        
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_xmax(self.get_fortran_address())["xmax"]
    
    @property
    def ymin(self):
        """
        
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_ymin(self.get_fortran_address())["ymin"]
    
    @property
    def ymax(self):
        """
        
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_ymax(self.get_fortran_address())["ymax"]
    
    @property
    def scal(self):
        """
        
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_scal(self.get_fortran_address())["scal"]
    
    @property
    def monitor(self):
        """
        
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_monitor(self.get_fortran_address())["monitor"]
    
    @property
    def norm_mon(self):
        """
        
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_norm_mon(self.get_fortran_address())["norm_mon"]
    
    @property
    def col_time(self):
        """
        
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_col_time(self.get_fortran_address())["col_time"]
    
    @property
    def step(self):
        """
        
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_step(self.get_fortran_address())["step"]
    
    @property
    def zerop(self):
        """
        
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_zerop(self.get_fortran_address())["zerop"]
    
    @property
    def Tsamp(self):
        """
        Sample Temperature
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_Tsamp(self.get_fortran_address())["Tsamp"]
    
    @property
    def Tset(self):
        """
        Setting Temperature (wished temperature)
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_Tset(self.get_fortran_address())["Tset"]
    
    @property
    def npts(self):
        """
        Number of points
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_npts(self.get_fortran_address())["npts"]
    
    def is_ct_step(self):
        """
        Constant step
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_ct_step(self.get_fortran_address())["ct_step"]
    
    def is_gy(self):
        """
        logicals for graphics
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_gy(self.get_fortran_address())["gy"]
    
    def is_gycalc(self):
        """
        logicals for graphics
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_gycalc(self.get_fortran_address())["gycalc"]
    
    def is_gbgr(self):
        """
        logicals for graphics
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_gbgr(self.get_fortran_address())["gbgr"]
    
    def is_gsigma(self):
        """
        logicals for graphics
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_gsigma(self.get_fortran_address())["gsigma"]
    
    def is_sig_var(self):
        """
        If .true. the content of sigma is in fact the variance
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_sig_var(self.get_fortran_address())["sig_var"]
    
    def is_al_x(self):
        """
        logicals for allocation
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_al_x(self.get_fortran_address())["al_x"]
    
    def is_al_y(self):
        """
        logicals for allocation
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_al_y(self.get_fortran_address())["al_y"]
    
    def is_al_ycalc(self):
        """
        logicals for allocation
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_al_ycalc(self.get_fortran_address())["al_ycalc"]
    
    def is_al_bgr(self):
        """
        logicals for allocation
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_al_bgr(self.get_fortran_address())["al_bgr"]
    
    def is_al_sigma(self):
        """
        logicals for allocation
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_al_sigma(self.get_fortran_address())["al_sigma"]
    
    def is_al_istat(self):
        """
        logicals for allocation
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_al_istat(self.get_fortran_address())["al_istat"]
    
    @property
    def conv(self):
        """
        Wavelengths or Dtt1, Dtt2 for converting to Q,d, etc
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_conv(self.get_fortran_address())["conv"]
    
    @property
    def x(self):
        """
        Scattering variable (2theta...)
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_x(self.get_fortran_address())["x"]
    
    @property
    def y(self):
        """
         Experimental intensity
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_y(self.get_fortran_address())["y"]
    
    @property
    def sigma(self):
        """
        observations sigma or variance (depending on sig_var)
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_sigma(self.get_fortran_address())["sigma"]
    
    @property
    def istat(self):
        """
        Information about the point 'i'
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_istat(self.get_fortran_address())["istat"]
    
    @property
    def ycalc(self):
        """
        Calculated intensity
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_ycalc(self.get_fortran_address())["ycalc"]
    
    @property
    def bgr(self):
        """
        Background
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_bgr(self.get_fortran_address())["bgr"]
    
    @property
    def nd(self):
        """
        Number of detectors contributing to the point 'i'
        """
        return CFML_api.crysfml_api.diffraction_patterns_get_nd(self.get_fortran_address())["nd"]


class PseudoVoigt():
    def __init__(self):
        self.fwhm = None
        self.eta  = None

        
    # Vp(x,f) = eta * L(x,f) + (1-eta)*G(x,f)
    def pseudovoigt_param(self, fwhm_G, fwhm_L):
    
        o1 = 2.69269
        o2 = 2.42843
        o3 = 4.47163
        o4 = 0.07842

        e1 = 1.36603
        e2 = 0.47719
        e3 = 0.11116

        # f=[f_{G}^{5}+2.69269f_{G}^{4}f_{L}+2.42843f_{G}^{3}f_{L}^{2}
        #       + 4.47163f_{G}^{2}f_{L}^{3}+0.07842f_{G}f_{L}^{4}+f_{L}^{5}]^{1/5

        self.fwhm = (fwhm_G**5.0 + o1*fwhm_G**4.0*fwhm_L + o2*fwhm_G**3.0*fwhm_L**2.0
                + o3*fwhm_G**2.0*fwhm_L**3.0+ o4*fwhm_G*fwhm_L**4.0+fwhm_L**5.0)**0.2

        #\eta =1.36603(f_{L}/f)-0.47719(f_{L}/f)^{2}+0.11116(f_{L}/f)^{3}
        ratio = fwhm_L/fwhm
        self.eta = max(1.0e-06, e1 * ratio - e2 * ratio**2 + e3 * ratio**3)

    def value(self,x):
        x2=x*x
        ag= 0.93943727869965133377234032841018/self.fwhm
        bg= 2.7725887222397812376689284858327/(self.fwhm**2)
        al= 0.63661977236758134307553505349006/self.fwhm
        bl= 4.0/self.fwhm**2
        gauss = ag* exp(-bg*x2)
        lor   = al/(1.0+bl*x2)
        pv_val = self.eta*lor + (1.0 - self.eta)*gauss

        return pv_val
