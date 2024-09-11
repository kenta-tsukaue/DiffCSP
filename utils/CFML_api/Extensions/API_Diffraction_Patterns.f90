! **************************************************************************
!
! CrysFML API
!
! @file      Src/Extensions/API_Diffraction_Patterns.f90
! @brief     Diffraction Patterns utilities based on CrysFML
!
! @homepage  https://code.ill.fr/scientific-software/crysfml
! @license   GNU LGPL (see LICENSE)
! @copyright Institut Laue Langevin 2020-now
! @authors   Scientific Computing Group at ILL (see AUTHORS), based on Elias Rabel work for Forpy
!
! **************************************************************************

module API_Diffraction_Patterns

  use forpy_mod
  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env

  use CFML_GlobalDeps,                only: cp, sp, pi
  
  use CFML_Diffraction_Patterns, only: &
       Diffraction_Pattern_Type

  use CFML_API_Calc_Powder_Pattern, only: &
       Calc_Powder_Pattern

  use API_reflections_utilities, only: &
       Reflection_List_type_p, &
       get_reflection_list_from_arg

  use API_IO_Formats, only: &
       Job_Info_type_p, &
       get_job_info_type_from_arg
  
  implicit none

  type Diffraction_Pattern_type_p
     type(Diffraction_Pattern_type), pointer :: p
  end type Diffraction_Pattern_type_p


contains

!!$  subroutine get_powpat_from_args(args, pow_pat_sim_c_p, indx)
!!$    type(tuple)                            :: args
!!$    type(Powder_Pattern_Simulation_Conditions_type_p), intent(out)  :: pow_pat_sim_c_p
!!$    integer, optional                      :: indx
!!$
!!$    type(object) :: arg_obj
!!$    type(dict) :: arg_dict
!!$
!!$    integer :: ierror
!!$    integer :: ii
!!$
!!$    if (present(indx)) then
!!$       ierror = args%getitem(arg_obj, indx)
!!$    else
!!$       ierror = args%getitem(arg_obj, 0)
!!$    endif
!!$
!!$    ierror = cast(arg_dict, arg_obj)
!!$
!!$    ierror = arg_dict%getitem(pow_pat_sim_c_p%p%job   , "job")
!!$    ierror = arg_dict%getitem(pow_pat_sim_c_p%p%Lambda, "lambda")
!!$    ierror = arg_dict%getitem(pow_pat_sim_c_p%p%U     , "u_resolution")
!!$    ierror = arg_dict%getitem(pow_pat_sim_c_p%p%V     , "v_resolution")
!!$    ierror = arg_dict%getitem(pow_pat_sim_c_p%p%W     , "w_resolution")
!!$    ierror = arg_dict%getitem(pow_pat_sim_c_p%p%X     , "x_resolution")
!!$    ierror = arg_dict%getitem(pow_pat_sim_c_p%p%Ls    , "lorentzian_size")
!!$    ierror = arg_dict%getitem(pow_pat_sim_c_p%p%Thmin , "theta_min")
!!$    ierror = arg_dict%getitem(pow_pat_sim_c_p%p%Thmax , "theta_max")
!!$    ierror = arg_dict%getitem(pow_pat_sim_c_p%p%step  , "theta_step")
!!$    ierror = arg_dict%getitem(pow_pat_sim_c_p%p%bkg   , "background")
!!$
!!$    call arg_obj%destroy
!!$    call arg_dict%destroy
!!$
!!$  end subroutine get_powpat_from_args
!!$
!!$
  subroutine get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer, indx)
    type(tuple)                            :: args
    type(Diffraction_Pattern_type_p), intent(out)  :: diffraction_pattern_type_pointer
    integer, optional                      :: indx

    type(object) :: arg_obj
    type(list) :: arg_list
    integer :: diffraction_pattern_type_p12(12)

    integer :: ierror
    integer :: ii
    type(object) :: t

    if (present(indx)) then
       ierror = args%getitem(arg_obj, indx)
    else
       ierror = args%getitem(arg_obj, 0)
    endif

    ierror = cast(arg_list, arg_obj)
    do ii=1,12
       ierror = arg_list%getitem(t, ii-1)
       ierror = cast(diffraction_pattern_type_p12(ii), t)
       call t%destroy
    enddo
    diffraction_pattern_type_pointer = transfer(diffraction_pattern_type_p12, diffraction_pattern_type_pointer)

    call arg_obj%destroy
    call arg_list%destroy

  end subroutine get_diffraction_pattern_type_from_arg

  function diffraction_patterns_del_powder_pattern(self_ptr, args_ptr) result(r) bind (c)
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "del_powder_pattern expects exactly 1 argument")
       call args%destroy
       return
    endif

    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer, 0)
    deallocate(diffraction_pattern_type_pointer%p%x)
    deallocate(diffraction_pattern_type_pointer%p%y)
    deallocate(diffraction_pattern_type_pointer%p%sigma)
    deallocate(diffraction_pattern_type_pointer%p%istat)
    deallocate(diffraction_pattern_type_pointer%p%ycalc)
    deallocate(diffraction_pattern_type_pointer%p%bgr)
    deallocate(diffraction_pattern_type_pointer%p%nd)
    deallocate(diffraction_pattern_type_pointer%p)

    ierror = dict_create(retval)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_del_powder_pattern

  function diffraction_patterns_compute_powder_pattern(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr)        :: r
    type(tuple)        :: args
    type(dict)         :: retval
    integer            :: num_args
    integer            :: ierror
    integer            :: ii

    type(list)   :: index_obj
    type(object) :: arg_obj

    type(Reflection_List_type_p)     :: hkl_p
    type(Diffraction_Pattern_type_p) :: pattern_p
    
    type(Job_info_type_p)            :: job_p
    integer                          :: pattern_p12(12)

    real(kind=cp)                    :: scalef

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)

    if (num_args /= 3) then
       call raise_exception(TypeError, "Calc_Powder_Pattern expects exactly 3 arguments")
       !Job_info, Reflectionlist, Scalef
       call args%destroy
       call arg_obj%destroy
       call index_obj%destroy
       return
    endif

    call get_job_info_type_from_arg(args, job_p, 0)
    
    call get_reflection_list_from_arg(args, hkl_p, 1)
    
    ierror = args%getitem(arg_obj, 2)
    ierror = cast_nonstrict(scalef, arg_obj)
    
    allocate(pattern_p%p)
    if ((job_p%p%patt_typ(1) == "XRAY_2THE") .or. (job_p%p%patt_typ(1) == "NEUT_2THE")) then
       call Calc_Powder_Pattern(job_p%p,scalef,hkl_p%p,pattern_p%p)
    else
       call raise_exception(TypeError, "Only 2THETA calculations are possible")
    endif
       
    pattern_p12 = transfer(pattern_p,pattern_p12)
    ierror = list_create(index_obj)
    do ii=1,12
       ierror = index_obj%append(pattern_p12(ii))
    end do
    ierror = dict_create(retval)
    ierror = retval%setitem("address", index_obj)

    r = retval%get_c_ptr()

    call args%destroy
    call arg_obj%destroy
    call index_obj%destroy

  end function diffraction_patterns_compute_powder_pattern


  function diffraction_patterns_get_title(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_title expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("title", trim(diffraction_pattern_type_pointer%p%title))

    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_title


  function diffraction_patterns_get_diff_kind(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_diff_kind expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("diff_kind", trim(diffraction_pattern_type_pointer%p%diff_kind))

    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_diff_kind


  function diffraction_patterns_get_scat_var(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_scat_var expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("scat_var", trim(diffraction_pattern_type_pointer%p%scat_var))

    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_scat_var


  function diffraction_patterns_get_xax_text(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_xax_text expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("xax_text", trim(diffraction_pattern_type_pointer%p%xax_text))

    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_xax_text


  function diffraction_patterns_get_yax_text(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_yax_text expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("yax_text", trim(diffraction_pattern_type_pointer%p%yax_text))

    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_yax_text


  function diffraction_patterns_get_instr(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_instr expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("instr", trim(diffraction_pattern_type_pointer%p%instr))

    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_instr


  function diffraction_patterns_get_filename(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_filename expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("filename", trim(diffraction_pattern_type_pointer%p%filename))

    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_filename


  function diffraction_patterns_get_filepath(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_filepath expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("filepath", trim(diffraction_pattern_type_pointer%p%filepath))

    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_filepath


  function diffraction_patterns_get_xmin(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_xmin expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("xmin", diffraction_pattern_type_pointer%p%xmin)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_xmin


  function diffraction_patterns_get_xmax(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_xmax expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("xmax", diffraction_pattern_type_pointer%p%xmax)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_xmax


  function diffraction_patterns_get_ymin(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_ymin expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("ymin", diffraction_pattern_type_pointer%p%ymin)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_ymin


  function diffraction_patterns_get_ymax(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_ymax expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("ymax", diffraction_pattern_type_pointer%p%ymax)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_ymax


  function diffraction_patterns_get_scal(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_scal expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("scal", diffraction_pattern_type_pointer%p%scal)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_scal


  function diffraction_patterns_get_monitor(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_monitor expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("monitor", diffraction_pattern_type_pointer%p%monitor)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_monitor


  function diffraction_patterns_get_norm_mon(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_norm_mon expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("norm_mon", diffraction_pattern_type_pointer%p%norm_mon)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_norm_mon


  function diffraction_patterns_get_col_time(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_col_time expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("col_time", diffraction_pattern_type_pointer%p%col_time)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_col_time


  function diffraction_patterns_get_step(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_step expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("step", diffraction_pattern_type_pointer%p%step)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_step


  function diffraction_patterns_get_zerop(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_zerop expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("zerop", diffraction_pattern_type_pointer%p%zerop)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_zerop


  function diffraction_patterns_get_Tsamp(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_Tsamp expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("Tsamp", diffraction_pattern_type_pointer%p%Tsamp)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_Tsamp


  function diffraction_patterns_get_Tset(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_Tset expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("Tset", diffraction_pattern_type_pointer%p%Tset)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_Tset


  function diffraction_patterns_get_npts(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_npts expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("npts", diffraction_pattern_type_pointer%p%npts)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_npts


  function diffraction_patterns_get_ct_step(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_ct_step expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("ct_step", diffraction_pattern_type_pointer%p%ct_step)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_ct_step


  function diffraction_patterns_get_gy(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_gy expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("gy", diffraction_pattern_type_pointer%p%gy)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_gy


  function diffraction_patterns_get_gycalc(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_gycalc expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("gycalc", diffraction_pattern_type_pointer%p%gycalc)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_gycalc


  function diffraction_patterns_get_gbgr(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_gbgr expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("gbgr", diffraction_pattern_type_pointer%p%gbgr)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_gbgr


  function diffraction_patterns_get_gsigma(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_gsigma expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("gsigma", diffraction_pattern_type_pointer%p%gsigma)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_gsigma


  function diffraction_patterns_get_sig_var(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_sig_var expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("sig_var", diffraction_pattern_type_pointer%p%sig_var)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_sig_var


  function diffraction_patterns_get_al_x(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_al_x expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("al_x", diffraction_pattern_type_pointer%p%al_x)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_al_x


  function diffraction_patterns_get_al_y(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_al_y expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("al_y", diffraction_pattern_type_pointer%p%al_y)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_al_y


  function diffraction_patterns_get_al_ycalc(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_al_ycalc expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("al_ycalc", diffraction_pattern_type_pointer%p%al_ycalc)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_al_ycalc


  function diffraction_patterns_get_al_bgr(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_al_bgr expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("al_bgr", diffraction_pattern_type_pointer%p%al_bgr)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_al_bgr


  function diffraction_patterns_get_al_sigma(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_al_sigma expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("al_sigma", diffraction_pattern_type_pointer%p%al_sigma)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_al_sigma


  function diffraction_patterns_get_al_istat(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_al_istat expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("al_istat", diffraction_pattern_type_pointer%p%al_istat)
    r = retval%get_c_ptr()
    call args%destroy

  end function diffraction_patterns_get_al_istat


  function diffraction_patterns_get_conv(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    type(ndarray) :: conv

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_conv expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = ndarray_create(conv, diffraction_pattern_type_pointer%p%conv)
    ierror = dict_create(retval)
    ierror = retval%setitem("conv", conv)
    r = retval%get_c_ptr()
    call args%destroy
    call conv%destroy

  end function diffraction_patterns_get_conv


  function diffraction_patterns_get_x(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    type(ndarray) :: x

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_x expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = ndarray_create(x, diffraction_pattern_type_pointer%p%x)
    ierror = dict_create(retval)
    ierror = retval%setitem("x", x)
    r = retval%get_c_ptr()
    call args%destroy
    call x%destroy

  end function diffraction_patterns_get_x


  function diffraction_patterns_get_y(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    type(ndarray) :: y

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_y expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = ndarray_create(y, diffraction_pattern_type_pointer%p%y)
    ierror = dict_create(retval)
    ierror = retval%setitem("y", y)
    r = retval%get_c_ptr()
    call args%destroy
    call y%destroy

  end function diffraction_patterns_get_y


  function diffraction_patterns_get_sigma(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    type(ndarray) :: sigma

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_sigma expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = ndarray_create(sigma, diffraction_pattern_type_pointer%p%sigma)
    ierror = dict_create(retval)
    ierror = retval%setitem("sigma", sigma)
    r = retval%get_c_ptr()
    call args%destroy
    call sigma%destroy

  end function diffraction_patterns_get_sigma


  function diffraction_patterns_get_istat(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    type(ndarray) :: istat

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_istat expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = ndarray_create(istat, diffraction_pattern_type_pointer%p%istat)
    ierror = dict_create(retval)
    ierror = retval%setitem("istat", istat)
    r = retval%get_c_ptr()
    call args%destroy
    call istat%destroy

  end function diffraction_patterns_get_istat


  function diffraction_patterns_get_ycalc(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    type(ndarray) :: ycalc

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_ycalc expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = ndarray_create(ycalc, diffraction_pattern_type_pointer%p%ycalc)
    ierror = dict_create(retval)
    ierror = retval%setitem("ycalc", ycalc)
    r = retval%get_c_ptr()
    call args%destroy
    call ycalc%destroy

  end function diffraction_patterns_get_ycalc


  function diffraction_patterns_get_bgr(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    type(ndarray) :: bgr

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_bgr expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = ndarray_create(bgr, diffraction_pattern_type_pointer%p%bgr)
    ierror = dict_create(retval)
    ierror = retval%setitem("bgr", bgr)
    r = retval%get_c_ptr()
    call args%destroy
    call bgr%destroy

  end function diffraction_patterns_get_bgr


  function diffraction_patterns_get_nd(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Diffraction_Pattern_Type_p) :: diffraction_pattern_type_pointer

    type(ndarray) :: nd

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_nd expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_diffraction_pattern_type_from_arg(args, diffraction_pattern_type_pointer)
    ierror = ndarray_create(nd, diffraction_pattern_type_pointer%p%nd)
    ierror = dict_create(retval)
    ierror = retval%setitem("nd", nd)
    r = retval%get_c_ptr()
    call args%destroy
    call nd%destroy

  end function diffraction_patterns_get_nd

end module API_Diffraction_Patterns
