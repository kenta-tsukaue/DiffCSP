# **************************************************************************
#
# CrysFML API
#
# @file      Src/PowderPatternSimulation.py
# @brief     Powder pattern simulation based on CrysFML
#
# @homepage  https://code.ill.fr/scientific-software/crysfml
# @license   GNU LGPL (see LICENSE)
# @copyright Institut Laue Langevin 2020-now
# @authors   Scientific Computing Group at ILL (see AUTHORS)
#
# **************************************************************************

import numpy as np

class PowderPatternSimulationSource():
    XRays = 0
    Neutrons = 1
    
class PowderPatternSimulationConditions():
    def __init__(self):
        self.title="Default Powder Pattern"
        self.lamb = 1.54056
        self.u_resolution = 0.0002
        self.v_resolution = -0.0002
        self.w_resolution = 0.012
        self.x_resolution = 0.0015
        self.theta_min = 1.00
        self.theta_step = 0.05
        self.theta_max = 135.0 # or 120 ? #int(2.0*asind(stlmax*1.54056))
        self.job = PowderPatternSimulationSource.Neutrons
        self.lorentzian_size = 1900.0
        self.bkg = 50.0
    
    def to_dict(self):
        return {
        "lambda":self.lamb,
        "u_resolution":self.u_resolution,
        "v_resolution":self.v_resolution,
        "w_resolution":self.w_resolution,
        "x_resolution":self.x_resolution,
        "theta_min":self.theta_min,
        "theta_step":self.theta_step,
        "theta_max":self.theta_max,
        "job":self.job,
        "lorentzian_size":self.lorentzian_size,
        "background":self.bkg
        }
        
    def getSinThetaOverLambdaMax(self):
        angle_max = min((self.theta_max+10.0)*0.5,90.0) / 180 * np.pi
        return np.sin(angle_max)/self.lamb
    
    def readFromCFLFile(self, file_name):
        with(open(file_name, "r")) as f:
            for line in f.readlines():
                line_splitted = line.split()
                if len(line_splitted):
                    if line_splitted[0].upper() == "TITLE":
                        self.title = line[line.index(line_splitted[1]):][:-1]
                    elif line_splitted[0].upper() == "UVWX":
                        self.u_resolution = float(line_splitted[1])
                        self.v_resolution = float(line_splitted[2])
                        self.w_resolution = float(line_splitted[3])
                        self.x_resolution = float(line_splitted[4])
                    elif line_splitted[0].upper() == "LAMBDA":
                        self.lamb = float(line_splitted[1])
                    elif line_splitted[0].upper() == "BACKGD":
                        self.bkg = float(line_splitted[1])
                    elif line_splitted[0].upper() == "JOBTYPE":
                        if line_splitted[1].upper() == "N":
                            self.job = PowderPatternSimulationSource.Neutrons
                        else:
                            self.job = PowderPatternSimulationSource.XRays
                    elif line_splitted[0].upper() == "PATTERN":
                        self.theta_min = float(line_splitted[1])
                        self.theta_step = float(line_splitted[2])
                        self.theta_max = float(line_splitted[3]) 
                    elif line_splitted[0].upper() == "SIZE_LG":
                        self.lorentzian_size = float(line_splitted[1])
    
    def __str__(self):
        ret = "TITLE: " + self.title + "\n"
        ret += "LAMBDA: " + str(self.lamb) + "\n"
        ret += "U RESOLUTION: " + str(self.u_resolution) + "\n"
        ret += "V RESOLUTION: " + str(self.v_resolution) + "\n"
        ret += "W RESOLUTION: " + str(self.w_resolution) + "\n"
        ret += "X RESOLUTION: " + str(self.x_resolution) + "\n"
        ret += "THETA MIN: " + str(self.theta_min) + "\n"
        ret += "THETA STEP: " + str(self.theta_step) + "\n"
        ret += "THETA MAX: " + str(self.theta_max) + "\n"
        if self.job == PowderPatternSimulationSource.Neutrons:
            ret += "JOB: NEUTRONS\n"
        else:
            ret += "JOB: XRAYS\n"
        ret += "LORENTZIAN SIZE: " + str(self.lorentzian_size) + "\n"
        ret += "BACKGROUND: " + str(self.bkg) + "\n"
        return ret

        
class PowderPatternSimulator():       
    def __init__(self):
        self.y = None
        self.x = None

    def compute(self, file, simulation_conditions=None):
        # Check if file exists
        file_exists = False
        if file[-4:] == ".cif" or file[-4:] == ".cfl":
            if os.path.exists(file):
                file_exists = True
                if file[-4:] == ".cif":
                    mode = "CIF"
                else:
                    mode = "CFL"
        elif os.path.exists(file+".cfl"):
                file = file + ".cfl"
                file_exists = True
                mode = "CFL"
        elif os.path.exists(file+".cif"):
                file = file + ".cif"
                file_exists = True
                mode = "CIF"
        
        if not file_exists:
            raise IOError("No file with name: " + file)
        else:
            if mode == "CIF" and simulation_conditions is None:
                logging.warning("No PowerPatternSimulationConditions provided, use default ones with CIF file")
                simulation_conditions = PowderPatternSimulationConditions()
                
            if mode == "CFL" and (not simulation_conditions is None):
                logging.warning("Use provided PowerPatternSimulationConditions instead of the ones within the CFL file")
                mode = "CF2"
            elif mode == "CFL":
                simulation_conditions = PowderPatternSimulationConditions()
                
            # Compute powder pattern
            ret = crysfml_binding.crysfml_powder_compute_powder_pattern(file,
                                                                          mode,
                                                                          simulation_conditions.title,
                                                                          simulation_conditions.lamb,
                                                                          simulation_conditions.u_resolution,
                                                                          simulation_conditions.v_resolution,
                                                                          simulation_conditions.w_resolution,
                                                                          simulation_conditions.x_resolution,
                                                                          simulation_conditions.theta_min,
                                                                          simulation_conditions.theta_step,
                                                                          simulation_conditions.theta_max,
                                                                          simulation_conditions.job,
                                                                          simulation_conditions.lorentzian_size,
                                                                          simulation_conditions.bkg)
            # Get back data 
            self.y = ret["Powder pattern"]#.copy()
            self.x = np.arange(ret["xmin"], ret["xmax"]+ret["xstep"]/2, ret["xstep"])
