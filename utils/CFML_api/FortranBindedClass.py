# **************************************************************************
#
# FortranBindedClass
#
# @file      Src/FortranBindedClass.py
# @brief     FortranBindedClass
#
# @homepage  https://code.ill.fr/scientific-software/crysfml
# @license   GNU LGPL (see LICENSE)
# @copyright Institut Laue Langevin 2020-now
# @authors   Scientific Computing Group at ILL (see AUTHORS)
#
# **************************************************************************

class FortranBindedClass():
    def __init__(self):
        self.__address = None
    
    @classmethod    
    def from_fortran_address(cls, address):
        instance = cls()
        instance._set_fortran_address(address)
        return instance
    
    def set_parent(self, parent):
        self.parent = parent
    
    def _set_fortran_address(self, address):
        self.__address = address

    def get_fortran_address(self):
        if self.__address:
            return self.__address.copy()
        else:
            return None