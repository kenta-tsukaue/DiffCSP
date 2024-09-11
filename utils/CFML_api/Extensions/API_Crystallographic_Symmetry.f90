! **************************************************************************
!
! CrysFML API
!
! @file      Src/Extensions/API_Crystallographic_Symmetry.f90
! @brief     Symmetry groups Fortran binding
!
! @homepage  https://code.ill.fr/scientific-software/crysfml
! @license   GNU LGPL (see LICENSE)
! @copyright Institut Laue Langevin 2020-now
! @authors   Scientific Computing Group at ILL (see AUTHORS), based on Elias Rabel work for Forpy
!
! **************************************************************************

module API_Crystallographic_Symmetry
  use forpy_mod
  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env

  use CFML_GlobalDeps,       only : Cp
  use CFML_Crystallographic_Symmetry, only: &
       Space_Group_Type, &
       set_SpaceGroup, &
       Write_SpaceGroup, &
       Sym_Oper_Type, &
       Wyckoff_Type, &
       Wyck_Pos_Type, &
       Get_Multip_Pos, &
       Get_Occ_Site

  implicit none

  type Space_Group_Type_p
     type(Space_Group_Type), pointer :: p
  end type Space_Group_Type_p

  type Symmetry_Operator_Type_p
     type(Sym_Oper_Type), pointer :: p
  end type Symmetry_Operator_Type_p

  type Wyckoff_Type_p
     type(Wyckoff_Type), pointer :: p
  end type Wyckoff_Type_p

  type Wyckoff_Orbit_Type_p
     type(Wyck_Pos_Type), pointer :: p
  end type Wyckoff_Orbit_Type_p

contains

  subroutine get_space_group_type_from_arg(args, spgr_p, indx)
    type(tuple)                            :: args
    type(Space_Group_type_p), intent(out)  :: spgr_p
    integer, optional                      :: indx
    
    type(object) :: arg_obj
    type(list) :: arg_list
    integer :: spgr_p12(12)
    
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
       ierror = cast(spgr_p12(ii), t)
       call t%destroy
    enddo
    spgr_p = transfer(spgr_p12, spgr_p)

    call arg_obj%destroy
    call arg_list%destroy

  end subroutine get_space_group_type_from_arg

  subroutine get_symmetry_operator_type_from_arg(args, symmetry_operator_p, indx)
    type(tuple)                            :: args
    type(Symmetry_Operator_Type_p), intent(out)  :: symmetry_operator_p
    integer, optional                      :: indx

    type(object) :: arg_obj
    type(list) :: arg_list
    integer :: symmetry_operator_p12(12)

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
       ierror = cast(symmetry_operator_p12(ii), t)
       call t%destroy
    enddo
    symmetry_operator_p = transfer(symmetry_operator_p12, symmetry_operator_p)

    call arg_obj%destroy
    call arg_list%destroy

  end subroutine get_symmetry_operator_type_from_arg

  subroutine get_wyckoff_type_from_arg(args, wyckoff_type_pointer, indx)
    type(tuple)                            :: args
    type(Wyckoff_Type_p), intent(out)  :: wyckoff_type_pointer
    integer, optional                      :: indx

    type(object) :: arg_obj
    type(list) :: arg_list
    integer :: wyckoff_type_p12(12)

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
       ierror = cast(wyckoff_type_p12(ii), t)
       call t%destroy
    enddo
    wyckoff_type_pointer = transfer(wyckoff_type_p12, wyckoff_type_pointer)

    call arg_obj%destroy
    call arg_list%destroy

  end subroutine get_wyckoff_type_from_arg

  subroutine get_wyckoff_orbit_type_from_arg(args, wyckoff_orbit_type_pointer, indx)
    type(tuple)                            :: args
    type(Wyckoff_Orbit_Type_p), intent(out)  :: wyckoff_orbit_type_pointer
    integer, optional                      :: indx

    type(object) :: arg_obj
    type(list) :: arg_list
    integer :: wyckoff_orbit_type_p12(12)

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
       ierror = cast(wyckoff_orbit_type_p12(ii), t)
       call t%destroy
    enddo
    wyckoff_orbit_type_pointer = transfer(wyckoff_orbit_type_p12, wyckoff_orbit_type_pointer)

    call arg_obj%destroy
    call arg_list%destroy

  end subroutine get_wyckoff_orbit_type_from_arg

  !-------------------------------------------------------------------------
  ! Implementation of our Python methods
  !-------------------------------------------------------------------------
  ! @brief
  function crystallographic_symmetry_set_spacegroup(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr)        :: r
    type(tuple)        :: args
    type(dict)         :: retval
    integer            :: num_args
    integer            :: ii
    integer            :: ierror

    !
    type(object)       :: spgr_obj
    type(list)         :: index_obj
    character(len=:), allocatable :: spgr
    type(Space_Group_type_p) :: spgr_p
    integer :: spgr_p12(12)

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple

    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "create_space_group expects exactly 1 argument")
       call args%destroy
       return
    endif
    ierror = args%getitem(spgr_obj, 0)
    ierror = cast_nonstrict(spgr, spgr_obj)

    !
    allocate(spgr_p%p)
    call set_spacegroup(spgr,spgr_p%p)
    spgr_p12 = transfer(spgr_p, spgr_p12)

    !
    ierror = list_create(index_obj)
    do ii=1,12
       ierror = index_obj%append(spgr_p12(ii))
    end do

    ierror = dict_create(retval)
    ierror = retval%setitem("address", index_obj)
    r = retval%get_c_ptr()

    deallocate(spgr)
    call args%destroy
    call index_obj%destroy


  end function crystallographic_symmetry_set_spacegroup

  function crystallographic_symmetry_del_spacegroup(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_type_p) :: spgr_p

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_description expects exactly 1 argument")
       call args%destroy
       return
    endif

    !
    call get_space_group_type_from_arg(args, spgr_p)

    if(allocated(spgr_p%p%Latt_trans)) deallocate(spgr_p%p%Latt_trans)
    if(allocated(spgr_p%p%SymOp)) deallocate(spgr_p%p%SymOp)
    if(allocated(spgr_p%p%SymOpSymb)) deallocate(spgr_p%p%SymOpSymb)
    deallocate(spgr_p%p)
    !
    ierror = dict_create(retval)
    r = retval%get_c_ptr()

    call args%destroy

  end function crystallographic_symmetry_del_spacegroup

  function crystallographic_symmetry_write_spacegroup(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_type_p) :: spgr_p

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_description expects exactly 1 argument")
       call args%destroy
       return
    endif

    !
    call get_space_group_type_from_arg(args, spgr_p)

    !
    call Write_SpaceGroup(spgr_p%p, full=.true.)

    !
    ierror = dict_create(retval)
    r = retval%get_c_ptr()

  end function crystallographic_symmetry_write_spacegroup


  function crystallographic_symmetry_get_SymOp(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ii
    type(list)         :: index_obj
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer
    type(Symmetry_Operator_Type_p) :: symmetry_operator_pointer
    integer :: symmetry_operator_pointer12(12)
    type(object) :: ind_obj
    integer :: ind

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "get_SymOp expects exactly 2 arguments")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)

    ierror = args%getitem(ind_obj, 1)
    ierror = cast(ind, ind_obj)

    symmetry_operator_pointer%p=>space_group_type_pointer%p%SymOp(ind)
    symmetry_operator_pointer12 = transfer(symmetry_operator_pointer, symmetry_operator_pointer12)
    ierror = list_create(index_obj)
    do ii=1,12
       ierror = index_obj%append(symmetry_operator_pointer12(ii))
    end do

    ierror = dict_create(retval)
    ierror = retval%setitem("SymOp", index_obj)
    r = retval%get_c_ptr()

    call args%destroy
    call index_obj%destroy
    call ind_obj%destroy

  end function crystallographic_symmetry_get_SymOp


  function crystallographic_symmetry_get_SymopSymb(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer
    type(object) :: ind_obj
    integer :: ind

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "get_SymopSymb expects exactly 2 arguments")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)
    ierror = args%getitem(ind_obj, 1)
    ierror = cast(ind, ind_obj)

    ierror = dict_create(retval)
    ierror = retval%setitem("SymopSymb", trim(space_group_type_pointer%p%SymopSymb(ind)))

    r = retval%get_c_ptr()

    call args%destroy
    call ind_obj%destroy

  end function crystallographic_symmetry_get_SymopSymb


  function crystallographic_symmetry_get_wyckoff(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    integer :: ii
    type(list)         :: index_obj
    type(Space_Group_Type_p) :: space_group_type_pointer
    type(Wyckoff_Type_p) :: wyckoff_pointer
    integer :: wyckoff_pointer12(12)

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_wyckoff expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)

    wyckoff_pointer%p=>space_group_type_pointer%p%Wyckoff
    wyckoff_pointer12 = transfer(wyckoff_pointer, wyckoff_pointer12)
    ierror = list_create(index_obj)
    do ii=1,12
       ierror = index_obj%append(wyckoff_pointer12(ii))
    end do

    ierror = dict_create(retval)
    ierror = retval%setitem("wyckoff", index_obj)
    r = retval%get_c_ptr()

    call args%destroy
    call index_obj%destroy

  end function crystallographic_symmetry_get_wyckoff

  function crystallographic_symmetry_get_latt_trans(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer

    type(ndarray) :: latt_trans

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_latt_trans expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)
    ierror = ndarray_create(latt_trans, space_group_type_pointer%p%latt_trans)
    ierror = dict_create(retval)
    ierror = retval%setitem("latt_trans", latt_trans)
    r = retval%get_c_ptr()
    call args%destroy
    call latt_trans%destroy

  end function crystallographic_symmetry_get_latt_trans


  function crystallographic_symmetry_get_hexa(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_hexa expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("hexa", space_group_type_pointer%p%hexa)
    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_hexa


  function crystallographic_symmetry_get_numspg(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_numspg expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("numspg", space_group_type_pointer%p%numspg)
    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_numspg


  function crystallographic_symmetry_get_spg_symb(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_spg_symb expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("spg_symb", trim(space_group_type_pointer%p%spg_symb))

    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_spg_symb


  function crystallographic_symmetry_get_hall(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_hall expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("hall", trim(space_group_type_pointer%p%hall))
    
    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_hall
  
  
  function crystallographic_symmetry_get_ghall(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_ghall expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("ghall", trim(space_group_type_pointer%p%ghall))
    
    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_ghall
  
  
  function crystallographic_symmetry_get_CrystalSys(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_CrystalSys expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("CrystalSys", trim(space_group_type_pointer%p%CrystalSys))
    
    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_CrystalSys
  
  
  function crystallographic_symmetry_get_Laue(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_Laue expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("Laue", trim(space_group_type_pointer%p%Laue))
    
    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_Laue
  
  
  function crystallographic_symmetry_get_PG(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_PG expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("PG", trim(space_group_type_pointer%p%PG))
    
    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_PG
  
  
  function crystallographic_symmetry_get_Info(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_Info expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("Info", trim(space_group_type_pointer%p%Info))
    
    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_Info
  
  
  function crystallographic_symmetry_get_SG_setting(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_SG_setting expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("SG_setting", trim(space_group_type_pointer%p%SG_setting))
    
    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_SG_setting
  
  
  function crystallographic_symmetry_get_SPG_lat(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_SPG_lat expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("SPG_lat", trim(space_group_type_pointer%p%SPG_lat))
    
    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_SPG_lat
  
  
  function crystallographic_symmetry_get_SPG_latsy(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_SPG_latsy expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("SPG_latsy", trim(space_group_type_pointer%p%SPG_latsy))
    
    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_SPG_latsy
  
  
  function crystallographic_symmetry_get_NumLat(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_NumLat expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("NumLat", space_group_type_pointer%p%NumLat)
    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_NumLat
  
  
  function crystallographic_symmetry_get_bravais(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_bravais expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("bravais", trim(space_group_type_pointer%p%bravais))
    
    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_bravais
  
  
  function crystallographic_symmetry_get_centre(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_centre expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("centre", trim(space_group_type_pointer%p%centre))
    
    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_centre
  
  
  function crystallographic_symmetry_get_centred(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_centred expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("centred", space_group_type_pointer%p%centred)
    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_centred
  
  
  function crystallographic_symmetry_get_centre_coord(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer
  
    type(ndarray) :: centre_coord
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_centre_coord expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)
    ierror = ndarray_create(centre_coord, space_group_type_pointer%p%centre_coord)
    ierror = dict_create(retval)
    ierror = retval%setitem("centre_coord", centre_coord)
    r = retval%get_c_ptr()
    call args%destroy
    call centre_coord%destroy

  end function crystallographic_symmetry_get_centre_coord
  
  
  function crystallographic_symmetry_get_numops(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_numops expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("numops", space_group_type_pointer%p%numops)
    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_numops
  
  
  function crystallographic_symmetry_get_multip(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_multip expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("multip", space_group_type_pointer%p%multip)
    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_multip
  
  
  function crystallographic_symmetry_get_num_gen(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_num_gen expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("num_gen", space_group_type_pointer%p%num_gen)
    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_num_gen
  

  function crystallographic_symmetry_get_R_asym_unit(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Space_Group_Type_p) :: space_group_type_pointer
  
    type(ndarray) :: R_asym_unit
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_R_asym_unit expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_space_group_type_from_arg(args, space_group_type_pointer)
    ierror = ndarray_create(R_asym_unit, space_group_type_pointer%p%R_asym_unit)
    ierror = dict_create(retval)
    ierror = retval%setitem("R_asym_unit", R_asym_unit)
    r = retval%get_c_ptr()
    call args%destroy
    call R_asym_unit%destroy

  end function crystallographic_symmetry_get_R_asym_unit

  function crystallographic_symmetry_get_symmetry_operator_rotation_matrix(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Symmetry_Operator_Type_p) :: symmetry_operator_pointer

    type(ndarray) :: rotation_matrix

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_symmetry_operator_rotation_matrix expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_symmetry_operator_type_from_arg(args, symmetry_operator_pointer)
    ierror = ndarray_create(rotation_matrix, symmetry_operator_pointer%p%Rot)
    ierror = dict_create(retval)
    ierror = retval%setitem("rotation_matrix", rotation_matrix)

    r = retval%get_c_ptr()
    call args%destroy
    call rotation_matrix%destroy

  end function crystallographic_symmetry_get_symmetry_operator_rotation_matrix

  function crystallographic_symmetry_get_symmetry_operator_trans_matrix(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Symmetry_Operator_Type_p) :: symmetry_operator_pointer

    type(ndarray) :: translation_matrix

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_symmetry_operator_translation_matrix expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_symmetry_operator_type_from_arg(args, symmetry_operator_pointer)
    ierror = ndarray_create(translation_matrix, symmetry_operator_pointer%p%Tr)
    ierror = dict_create(retval)
    ierror = retval%setitem("translation_matrix", translation_matrix)
    r = retval%get_c_ptr()
    call args%destroy
    call translation_matrix%destroy

  end function crystallographic_symmetry_get_symmetry_operator_trans_matrix

  function crystallographic_symmetry_get_wyckoff_num_orbit(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Wyckoff_Type_p) :: wyckoff_type_pointer

    type(ndarray) :: translation_matrix

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_symmetry_operator_translation_matrix expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_wyckoff_type_from_arg(args, wyckoff_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("num_orbit", wyckoff_type_pointer%p%num_orbit)
    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_wyckoff_num_orbit

  function crystallographic_symmetry_get_wyckoff_orbits(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ii
    type(list)         :: index_obj
    integer :: ierror
    type(Wyckoff_Type_p) :: wyckoff_type_pointer
    type(Wyckoff_Orbit_Type_p) :: wyckoff_orbit_type_pointer
    integer :: wyckoff_orbit_type_pointer12(12)
    type(object) :: ind_obj
    integer :: ind

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "get_wyckoff_orbits expects exactly 2 arguments")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_wyckoff_type_from_arg(args, wyckoff_type_pointer)

    ierror = args%getitem(ind_obj, 1)
    ierror = cast(ind, ind_obj)

    wyckoff_orbit_type_pointer%p=>wyckoff_type_pointer%p%orbit(ind)
    wyckoff_orbit_type_pointer12 = transfer(wyckoff_orbit_type_pointer, wyckoff_orbit_type_pointer12)
    ierror = list_create(index_obj)
    do ii=1,12
       ierror = index_obj%append(wyckoff_orbit_type_pointer12(ii))
    end do

    ierror = dict_create(retval)
    ierror = retval%setitem("orbit", index_obj)
    r = retval%get_c_ptr()

    call args%destroy
    call index_obj%destroy
    call ind_obj%destroy

  end function crystallographic_symmetry_get_wyckoff_orbits

  function crystallographic_symmetry_get_wyckoff_multp(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Wyckoff_Orbit_Type_p) :: wyckoff_orbit_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_wyckoff_multp expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_wyckoff_orbit_type_from_arg(args, wyckoff_orbit_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("multp", wyckoff_orbit_type_pointer%p%multp)
    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_wyckoff_multp


  function crystallographic_symmetry_get_wyckoff_site(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Wyckoff_Orbit_Type_p) :: wyckoff_orbit_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_wyckoff_site expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_wyckoff_orbit_type_from_arg(args, wyckoff_orbit_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("site", trim(wyckoff_orbit_type_pointer%p%site))

    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_wyckoff_site


  function crystallographic_symmetry_get_wyckoff_norb(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Wyckoff_Orbit_Type_p) :: wyckoff_orbit_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_wyckoff_norb expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_wyckoff_orbit_type_from_arg(args, wyckoff_orbit_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("norb", wyckoff_orbit_type_pointer%p%norb)
    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_wyckoff_norb


  function crystallographic_symmetry_get_wyckoff_str_orig(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Wyckoff_Orbit_Type_p) :: wyckoff_orbit_type_pointer

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_wyckoff_str_orig expects exactly 1 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_wyckoff_orbit_type_from_arg(args, wyckoff_orbit_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("str_orig", trim(wyckoff_orbit_type_pointer%p%str_orig))

    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_wyckoff_str_orig


  function crystallographic_symmetry_get_wyckoff_str_orbit(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Wyckoff_Orbit_Type_p) :: wyckoff_orbit_type_pointer
    type(object) :: ind_obj
    integer :: ind

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "get_wyckoff_str_orbit expects exactly 2 argument")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_wyckoff_orbit_type_from_arg(args, wyckoff_orbit_type_pointer)
    ierror = args%getitem(ind_obj, 1)
    ierror = cast(ind, ind_obj)
    ierror = dict_create(retval)
    ierror = retval%setitem("str_orbit", trim(wyckoff_orbit_type_pointer%p%str_orbit(ind)))

    r = retval%get_c_ptr()
    call args%destroy

  end function crystallographic_symmetry_get_wyckoff_str_orbit



  function crystallographic_symmetry_get_multip_pos_crys(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror

    type(Space_Group_type_p) :: spg
    type(object)                         :: pos_obj
    type(ndarray)                        :: pos_nd
    real(kind=cp), dimension(:), pointer :: pos_p

    integer :: mult
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "get_multip_pos_crys expects exactly 2 arguments")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    ierror = args%getitem(pos_obj, 0)        !pos -> x,y,z
    ierror = cast(pos_nd, pos_obj)
    ierror = pos_nd%get_data(pos_p)

    call get_space_group_type_from_arg(args, spg, 1)
    
    mult = Get_Multip_Pos(pos_p, spg%p)
    !write(*,*) "f90 mult",mult

    ierror = dict_create(retval)
    ierror = retval%setitem("mult", mult)
    
    r = retval%get_c_ptr()

    call args%destroy
    call pos_obj%destroy
    
    
  end function crystallographic_symmetry_get_multip_pos_crys


  function crystallographic_symmetry_get_occ_site(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror

    type(Space_Group_type_p) :: spg
    type(object)                         :: pos_obj
    type(ndarray)                        :: pos_nd
    real(kind=cp), dimension(:), pointer :: pos_p

    real(kind=cp) :: occ
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "get_multip_pos_crys expects exactly 2 arguments")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    ierror = args%getitem(pos_obj, 0)        !pos -> x,y,z
    ierror = cast(pos_nd, pos_obj)
    ierror = pos_nd%get_data(pos_p)

    call get_space_group_type_from_arg(args, spg, 1)
    
    occ= Get_Occ_site(pos_p, spg%p)
    !write(*,*) "f90 mult",mult

    ierror = dict_create(retval)
    ierror = retval%setitem("occ", occ)
    
    r = retval%get_c_ptr()

    call args%destroy
    call pos_obj%destroy
    
    
  end function crystallographic_symmetry_get_occ_site

end module API_Crystallographic_Symmetry
