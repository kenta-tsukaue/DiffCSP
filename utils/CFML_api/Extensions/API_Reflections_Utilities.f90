! **************************************************************************
!
! CrysFML API
!
! @file      Src/Extensions/API_IO_Formats.f90
! @brief     CFML IO Formats Fortran binding
!
! @homepage  https://code.ill.fr/scientific-software/crysfml
! @license   GNU LGPL (see LICENSE)
! @copyright Institut Laue Langevin 2020-now
! @authors   Scientific Computing Group at ILL (see AUTHORS), based on Elias Rabel work for Forpy
!
! **************************************************************************

module API_Reflections_Utilities
  
  use forpy_mod
  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env 

  use CFML_GlobalDeps,                 only: Cp
  
  use CFML_Reflections_Utilities,      only: &
       Reflection_Type, &
       Reflection_List_Type, &
       Hkl_uni, &
       get_maxnumref, &
       Write_RefList_Info
  
  use API_Crystallographic_Symmetry, only: &
       Space_Group_Type_p, &
       get_space_group_type_from_arg
  
  use API_Crystal_Metrics, only: &
       Crystal_Cell_Type_p, &
       get_cell_from_arg
  
  implicit none

  type Reflection_List_type_p
     type(Reflection_List_type), pointer :: p
  end type Reflection_List_type_p

  type Reflection_type_p
     type(Reflection_type), pointer :: p
  end type Reflection_type_p

contains

  subroutine get_reflection_list_from_arg(args, reflection_list_p, indx)
    type(tuple)                            :: args
    type(Reflection_List_type_p), intent(out) :: reflection_list_p
    integer, optional                      :: indx

    type(object) :: arg_obj
    type(list) :: arg_list
    integer :: reflection_list_p12(12)

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
       ierror = cast(reflection_list_p12(ii), t)
       call t%destroy
    enddo
    reflection_list_p = transfer(reflection_list_p12, reflection_list_p)
    call arg_obj%destroy
    call arg_list%destroy

  end subroutine get_reflection_list_from_arg

  function reflections_utilities_del_reflection_list(self_ptr, args_ptr) result(r) bind(c)
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Reflection_List_type_p)    :: hkl_p

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

    call get_reflection_list_from_arg(args, hkl_p, 0)
    if (allocated(hkl_p%p%ref)) deallocate(hkl_p%p%ref)
    deallocate(hkl_p%p)

    ierror = dict_create(retval)
    r = retval%get_c_ptr()
    call args%destroy
  end function reflections_utilities_del_reflection_list

  function reflections_utilities_get_nref(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Reflection_List_type_p) :: hkl_p
  
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_nrefs expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_reflection_list_from_arg(args, hkl_p)

    ierror = dict_create(retval)
    ierror = retval%setitem("nref", hkl_p%p%nref)
    r = retval%get_c_ptr()
    call args%destroy
    
     
  end function reflections_utilities_get_nref

   ! @brief Print the list of reflections and structure factors to standard output
  function reflections_utilities_write_reflist_info(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    
    type(tuple) :: args
    type(dict)  :: retval
    integer     :: num_args
    integer     :: ierror

    type(Reflection_List_Type_p)   :: reflist_p

    r = C_NULL_PTR      
    call unsafe_cast_from_c_ptr(args, args_ptr)
    
    ierror = args%len(num_args)
    
    if (num_args /= 1) then
       call raise_exception(TypeError, "write_reflist_info expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    call get_reflection_list_from_arg(args, reflist_p)

    call Write_RefList_Info(reflist_p%p)

    ierror = dict_create(retval)
    r = retval%get_c_ptr()

    call args%destroy

  end function reflections_utilities_write_reflist_info


  function reflections_utilities_hkl_uni_reflist(self_ptr, args_ptr) result(r) bind(c)

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

    type(Crystal_Cell_type_p)       :: cell_p
    type(Space_Group_type_p)        :: spg_p
    type(Reflection_List_type_p)    :: hkl_p
    integer                         :: hkl_p12(12)

    logical       :: friedel
    real(kind=cp) :: value1, value2
    integer       :: maxnumref
    integer, dimension(3,2) :: hlim
    logical :: strict_flag
    

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)

    if (num_args /= 5) then
       call raise_exception(TypeError, "hkl_uni_reflist expects exactly 5 arguments")
       !Cell, SpG, lFriedel, value1, value2)
       call args%destroy
       return
    endif

    call get_cell_from_arg(args, cell_p, 0)
    
    call get_space_group_type_from_arg(args, spg_p, 1)

    ierror = args%getitem(arg_obj, 2)
    ierror = cast_nonstrict(friedel, arg_obj)

    ierror = args%getitem(arg_obj, 3)
    ierror = cast_nonstrict(value1, arg_obj)

    ierror = args%getitem(arg_obj, 4)
    ierror = cast_nonstrict(value2, arg_obj)

    !> @todo here value2 is assumed to be stlmax 
    MaxNumRef = get_maxnumref(value2,Cell_p%p%CellVol,mult=2*SpG_p%p%NumOps)

    hlim(1,:) = [-2, 2]   ! H の範囲
    hlim(2,:) = [-2, 2]   ! K の範囲
    hlim(3,:) = [-2, 2]   ! L の範囲

    ! strict フラグを True に設定
    strict_flag = .true.

    allocate(hkl_p%p)
    call Hkl_Uni(cell_p%p,spg_p%p,friedel,value1,value2,"s",MaxNumRef,hkl_p%p, hlim=hlim, strict=strict_flag)

    hkl_p12 = transfer(hkl_p,hkl_p12)
    ierror = list_create(index_obj)
    do ii=1,12
       ierror = index_obj%append(hkl_p12(ii))
    end do

    ierror = dict_create(retval)
    ierror = retval%setitem("address", index_obj)
    r = retval%get_c_ptr()
    call args%destroy

  end function reflections_utilities_hkl_uni_reflist

  function reflections_utilities_get_item(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    
    type(tuple) :: args
    type(dict)  :: retval
    integer     :: num_args
    integer     :: ierror
    integer     :: ii

    type(Reflection_list_type_p)               :: alist_p
    type(Reflection_type_p)                    :: a_p

    type(object) :: item_obj
    integer      :: item, a_p12(12)
    type(list)   :: a_obj

    r = C_NULL_PTR      
    call unsafe_cast_from_c_ptr(args, args_ptr)
    
    ierror = args%len(num_args)
    
    if (num_args /= 2) then
       call raise_exception(TypeError, "get_item expects exactly 2 arguments")
       call args%destroy
       return
    endif
    
    call get_reflection_list_from_arg(args, alist_p, 0)

    ierror = args%getitem(item_obj, 1)
    ierror = cast_nonstrict(item, item_obj)
    allocate(a_p%p)
    a_p%p = alist_p%p%ref(item+1)

    a_p12    = transfer(a_p, a_p12)
    ierror = list_create(a_obj)
    do ii=1,12
       ierror = a_obj%append(a_p12(ii))
    end do

    ierror = dict_create(retval)
    ierror = retval%setitem("Ref", a_obj)

    r = retval%get_c_ptr()
    call args%destroy
    call item_obj%destroy
    call a_obj%destroy

  end function reflections_utilities_get_item

  !---------------------------------------
  ! Reflection type
  !---------------------------------------

    subroutine get_reflection_type_from_arg(args, reflection_p, indx)
    type(tuple)                            :: args
    type(Reflection_type_p), intent(out)   :: reflection_p
    integer, optional                      :: indx

    type(object) :: arg_obj
    type(list) :: arg_list
    integer :: reflection_p12(12)

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
       ierror = cast(reflection_p12(ii), t)
       call t%destroy
    enddo
    reflection_p = transfer(reflection_p12, reflection_p)
    call arg_obj%destroy
    call arg_list%destroy

  end subroutine get_reflection_type_from_arg

  function reflections_utilities_del_reflection(self_ptr, args_ptr) result(r) bind(c)
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Reflection_type_p)    :: hkl_p

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "del_reflection expects exactly 1 argument")
       call args%destroy
       return
    endif

    call get_reflection_type_from_arg(args, hkl_p, 0)
    deallocate(hkl_p%p)

    ierror = dict_create(retval)
    r = retval%get_c_ptr()
    call args%destroy
  end function reflections_utilities_del_reflection

  

  function reflections_utilities_get_H(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Reflection_type_p) :: reflection_type_pointer
  
    type(ndarray) :: H
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_H expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_reflection_type_from_arg(args, reflection_type_pointer)
    ierror = ndarray_create(H, reflection_type_pointer%p%H)
    ierror = dict_create(retval)
    ierror = retval%setitem("H", H)
    r = retval%get_c_ptr()
    call args%destroy
    call H%destroy
     
  end function reflections_utilities_get_H
  
  
  function reflections_utilities_get_Mult(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Reflection_type_p) :: reflection_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_Mult expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_reflection_type_from_arg(args, reflection_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("Mult", reflection_type_pointer%p%Mult)
    r = retval%get_c_ptr()
    call args%destroy
     
  end function reflections_utilities_get_Mult
  
  
  function reflections_utilities_get_Fo(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Reflection_type_p) :: reflection_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_Fo expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_reflection_type_from_arg(args, reflection_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("Fo", reflection_type_pointer%p%Fo)
    r = retval%get_c_ptr()
    call args%destroy
     
  end function reflections_utilities_get_Fo
  
  
  function reflections_utilities_get_Fc(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Reflection_type_p) :: reflection_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_Fc expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_reflection_type_from_arg(args, reflection_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("Fc", reflection_type_pointer%p%Fc)
    r = retval%get_c_ptr()
    call args%destroy
     
  end function reflections_utilities_get_Fc
  
  
  function reflections_utilities_get_SFo(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Reflection_type_p) :: reflection_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_SFo expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_reflection_type_from_arg(args, reflection_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("SFo", reflection_type_pointer%p%SFo)
    r = retval%get_c_ptr()
    call args%destroy
     
  end function reflections_utilities_get_SFo
  
  
  function reflections_utilities_get_S(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Reflection_type_p) :: reflection_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_S expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_reflection_type_from_arg(args, reflection_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("S", reflection_type_pointer%p%S)
    r = retval%get_c_ptr()
    call args%destroy
     
  end function reflections_utilities_get_S
  
  
  function reflections_utilities_get_W(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Reflection_type_p) :: reflection_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_W expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_reflection_type_from_arg(args, reflection_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("W", reflection_type_pointer%p%W)
    r = retval%get_c_ptr()
    call args%destroy
     
  end function reflections_utilities_get_W
  
  
  function reflections_utilities_get_Phase(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Reflection_type_p) :: reflection_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_Phase expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_reflection_type_from_arg(args, reflection_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("Phase", reflection_type_pointer%p%Phase)
    r = retval%get_c_ptr()
    call args%destroy
     
  end function reflections_utilities_get_Phase
  
  
  function reflections_utilities_get_A(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Reflection_type_p) :: reflection_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_A expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_reflection_type_from_arg(args, reflection_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("A", reflection_type_pointer%p%A)
    r = retval%get_c_ptr()
    call args%destroy
     
  end function reflections_utilities_get_A
  
  
  function reflections_utilities_get_B(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Reflection_type_p) :: reflection_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_B expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_reflection_type_from_arg(args, reflection_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("B", reflection_type_pointer%p%B)
    r = retval%get_c_ptr()
    call args%destroy
     
  end function reflections_utilities_get_B

 
  

end module API_Reflections_Utilities
