module API_Error_Messages

  use forpy_mod
  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env

  use CFML_Crystallographic_Symmetry,  only: ERR_Symm,     ERR_Symm_Mess
  use CFML_Crystal_Metrics,            only: ERR_Crys,     ERR_Crys_Mess
  use CFML_Atom_TypeDef,               only: ERR_Atmd,     ERR_Atmd_Mess
  use CFML_String_Utilities,           only: ERR_String,   ERR_String_Mess
  use CFML_Diffraction_Patterns,       only: ERR_DiffPatt, ERR_DiffPatt_Mess
  use CFML_IO_Formats,                 only: ERR_Form,     ERR_Form_Mess
  use CFML_Reflections_Utilities,      only: ERR_Refl,     ERR_Refl_Mess
  use CFML_Magnetic_Symmetry,          only: ERR_MagSym,   ERR_MagSym_Mess
  use CFML_Structure_Factors,          only: ERR_SFac,     ERR_SFac_Mess
  use CFML_Magnetic_Structure_Factors, only: ERR_MSFac,    ERR_MSFac_Mess

  implicit none

contains

  function error_messages(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r

    type(tuple) :: args
    type(dict)  :: retval
    integer     :: num_args
    integer     :: ierror

    r = C_NULL_PTR
    call unsafe_cast_from_c_ptr(args, args_ptr)

    ierror = args%len(num_args)

    if (num_args /= 0) then
       call raise_exception(TypeError, "get_item expects exactly 0 arguments")
       call args%destroy
       return
    endif

    ierror = dict_create(retval)
    ierror = retval%setitem("ERR_Symm", ERR_Symm)
    ierror = retval%setitem("ERR_Symm_Mess", ERR_Symm_Mess)
    ierror = retval%setitem("ERR_Crys", ERR_Crys)
    ierror = retval%setitem("ERR_Crys_Mess", ERR_Crys_Mess)
    ierror = retval%setitem("ERR_Atmd", ERR_Atmd)
    ierror = retval%setitem("ERR_Atmd_Mess", ERR_Atmd_Mess)
    ierror = retval%setitem("ERR_String", ERR_String)
    ierror = retval%setitem("ERR_String_Mess", ERR_String_Mess)
    ierror = retval%setitem("ERR_DiffPatt", ERR_DiffPatt)
    ierror = retval%setitem("ERR_DiffPatt_Mess", ERR_DiffPatt_Mess)
    ierror = retval%setitem("ERR_Form", ERR_Form)
    ierror = retval%setitem("ERR_Form_Mess", ERR_Form_Mess)
    ierror = retval%setitem("ERR_Refl", ERR_Refl)
    ierror = retval%setitem("ERR_Refl_Mess", ERR_Refl_Mess)
    ierror = retval%setitem("ERR_MagSym", ERR_MagSym)
    ierror = retval%setitem("ERR_MagSym_Mess", ERR_MagSym_Mess)
    ierror = retval%setitem("ERR_SFac", ERR_SFac)
    ierror = retval%setitem("ERR_SFac_Mess", ERR_SFac_Mess)
    ierror = retval%setitem("ERR_MSFac", ERR_MSFac)
    ierror = retval%setitem("ERR_MSFac_Mess", ERR_MSFac_Mess)

    r = retval%get_c_ptr()
    call args%destroy

  end function error_messages

  end module API_Error_Messages
