class classproperty(object):
    def __init__(self, f):
        self.f = f
    def __get__(self, obj, owner):
        return self.f(owner)
    
import CFML_api.crysfml_api
class ErrorMessages():
    @classproperty
    def ERR_Symm(cls):
        return CFML_api.crysfml_api.error_messages()["ERR_Symm"]
    
    @classproperty
    def ERR_Symm_Mess(cls):
        return CFML_api.crysfml_api.error_messages()["ERR_Symm_Mess"]
    
    @classproperty
    def ERR_Crys(cls):
        return CFML_api.crysfml_api.error_messages()["ERR_Crys"]
    
    @classproperty
    def ERR_Crys_Mess(cls):
        return CFML_api.crysfml_api.error_messages()["ERR_Crys_Mess"]
    
    @classproperty
    def ERR_Atmd(cls):
        return CFML_api.crysfml_api.error_messages()["ERR_Atmd"]
    
    @classproperty
    def ERR_Atmd_Mess(cls):
        return CFML_api.crysfml_api.error_messages()["ERR_Atmd_Mess"]
    
    @classproperty
    def ERR_String(cls):
        return CFML_api.crysfml_api.error_messages()["ERR_String"]
    
    @classproperty
    def ERR_String_Mess(cls):
        return CFML_api.crysfml_api.error_messages()["ERR_String_Mess"]
    
    @classproperty
    def ERR_DiffPatt(cls):
        return CFML_api.crysfml_api.error_messages()["ERR_DiffPatt"]
    
    @classproperty
    def ERR_DiffPatt_Mess(cls):
        return CFML_api.crysfml_api.error_messages()["ERR_DiffPatt_Mess"]
    
    @classproperty
    def ERR_Form(cls):
        return CFML_api.crysfml_api.error_messages()["ERR_Form"]
    
    @classproperty
    def ERR_Form_Mess(cls):
        return CFML_api.crysfml_api.error_messages()["ERR_Form_Mess"]
    
    @classproperty
    def ERR_Refl(cls):
        return CFML_api.crysfml_api.error_messages()["ERR_Refl"]
    
    @classproperty
    def ERR_Refl_Mess(cls):
        return CFML_api.crysfml_api.error_messages()["ERR_Refl_Mess"]
    
    @classproperty
    def ERR_MagSym(cls):
        return CFML_api.crysfml_api.error_messages()["ERR_MagSym"]
    
    @classproperty
    def ERR_MagSym_Mess(cls):
        return CFML_api.crysfml_api.error_messages()["ERR_MagSym_Mess"]
    
    @classproperty
    def ERR_SFac(cls):
        return CFML_api.crysfml_api.error_messages()["ERR_SFac"]
    
    @classproperty
    def ERR_SFac_Mess(cls):
        return CFML_api.crysfml_api.error_messages()["ERR_SFac_Mess"]
    
    @classproperty
    def ERR_MSFac(cls):
        return CFML_api.crysfml_api.error_messages()["ERR_MSFac"]
    
    @classproperty
    def ERR_MSFac_Mess(cls):
        return CFML_api.crysfml_api.error_messages()["ERR_MSFac_Mess"]
