! **************************************************************************
!
! CrysFML API
!
! @file      Src/Extensions/API_Crystal_Metrics.f90
! @brief     Crystall Cell Fortran binding
!
! @homepage  https://code.ill.fr/scientific-software/crysfml
! @license   GNU LGPL (see LICENSE)
! @copyright Institut Laue Langevin 2020-now
! @authors   Scientific Computing Group at ILL (see AUTHORS), based on Elias Rabel work for Forpy
!
! **************************************************************************

module API_Crystal_Metrics
  
  use forpy_mod
  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env

  use CFML_GlobalDeps,       only : Cp
  use CFML_Crystal_Metrics,  only: &
       Crystal_Cell_Type, &
       Write_Crystal_Cell, &
       Set_Crystal_Cell, &
       Cart_Vector, &
       Cart_U_Vector, &
       Get_Betas_from_Biso
  
  implicit none

  !type definitions 
  type Crystal_Cell_type_p
     type(Crystal_Cell_type), pointer :: p
  end type Crystal_Cell_type_p
  
contains 

  subroutine get_cell_from_arg(args, cell_p, indx)
    type(tuple)                            :: args
    type(Crystal_Cell_type_p), intent(out) :: cell_p
    integer, optional                      :: indx
    
    type(object) :: arg_obj
    type(list) :: arg_list
    integer :: cell_p12(12)
  
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
       ierror = cast(cell_p12(ii), t)
       call t%destroy
    enddo
    cell_p = transfer(cell_p12, cell_p)
    call arg_obj%destroy
    call arg_list%destroy

  end subroutine get_cell_from_arg

  
  ! @brief Create a crystal cell from a set of lattice parameters
  function crystal_metrics_set_crystal_cell(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    
    type(tuple) :: args
    type(dict)  :: retval
    integer     :: num_args
    integer     :: ierror
    integer     :: i,ii
    type(list)  :: index_obj

    type(object)                         :: cellv_obj, angl_obj
    type(ndarray)                        :: cellv_nd, angl_nd
    real(kind=cp), dimension(:), pointer :: cellv_p, angl_p
    type(Crystal_Cell_type_p)            :: cell_p
    integer                              :: cell_p12(12)
       
    r = C_NULL_PTR      
    call unsafe_cast_from_c_ptr(args, args_ptr)
    
    ierror = args%len(num_args)
    
    if (num_args < 2) then
       call raise_exception(TypeError, "set_crystal_Cell expects at least 2 arguments")
       call args%destroy
       return
    endif

    !> @todo optional arguments Cartype,Scell,Sangl
    if (num_args > 2) then
       call raise_exception(TypeError, "optional arguments not yet available for set_crystal_Cell")
       call args%destroy
       return
    endif
     
    ierror = args%getitem(cellv_obj, 0)        !cellv  In -> a,b,c
    ierror = cast(cellv_nd, cellv_obj)
    ierror = cellv_nd%get_data(cellv_p)
        
    ierror = args%getitem(angl_obj, 1)         !angl   In -> angles of cell parameters
    ierror = cast(angl_nd, angl_obj)
    ierror = angl_nd%get_data(angl_p)
       
    allocate(cell_p%p)
    call Set_Crystal_Cell(cellv_p, angl_p, cell_p%p)
    
    cell_p12 = transfer(cell_p,cell_p12)
    ierror = list_create(index_obj)
    do ii=1,12
       ierror = index_obj%append(cell_p12(ii))
    end do

    ierror = dict_create(retval)
    ierror = retval%setitem("address", index_obj)
    r = retval%get_c_ptr()

    call args%destroy
    call index_obj%destroy

    call args%destroy

  end function crystal_metrics_set_crystal_cell

  ! @brief Print the description of the cell to standard output
  function crystal_metrics_del_crystal_cell(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r

    type(tuple) :: args
    type(dict)  :: retval
    integer     :: num_args
    integer     :: ierror

    type(Crystal_Cell_type_p)   :: cell_p

    r = C_NULL_PTR
    call unsafe_cast_from_c_ptr(args, args_ptr)

    ierror = args%len(num_args)

    if (num_args /= 1) then
       call raise_exception(TypeError, "del_crystal_Cell expects exactly 1 argument")
       call args%destroy
       return
    endif

    call get_cell_from_arg(args, cell_p)

    deallocate(cell_p%p)

    ierror = dict_create(retval)
    r = retval%get_c_ptr()

    call args%destroy

  end function crystal_metrics_del_crystal_cell

  ! @brief Print the description of the cell to standard output
  function crystal_metrics_write_crystal_cell(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    
    type(tuple) :: args
    type(dict)  :: retval
    integer     :: num_args
    integer     :: ierror

    type(Crystal_Cell_type_p)   :: cell_p

    r = C_NULL_PTR      
    call unsafe_cast_from_c_ptr(args, args_ptr)
    
    ierror = args%len(num_args)
    
    if (num_args /= 1) then
       call raise_exception(TypeError, "write_crystal_Cell expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    call get_cell_from_arg(args, cell_p)

    call Write_Crystal_Cell(Cell_p%p)

    ierror = dict_create(retval)
    r = retval%get_c_ptr()

    call args%destroy

  end function crystal_metrics_write_crystal_cell

  
  ! @brief Get the parameter 'cell' from a Crystal_Cell_Type object
  function crystal_metrics_get_cell(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Crystal_Cell_type_p)   :: cell_p
  
    type(ndarray) :: cell
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_cell expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    call get_cell_from_arg(args, cell_p)
    ierror = ndarray_create(cell, cell_p%p%cell)
    ierror = dict_create(retval)
    ierror = retval%setitem("cell", cell)
    
    r = retval%get_c_ptr()
    
    call args%destroy

  end function crystal_metrics_get_cell


  ! @brief Get the parameter 'ang' from a Crystal_Cell_Type object
  function crystal_metrics_get_ang(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Crystal_Cell_type_p)   :: cell_p
    
    type(ndarray) :: ang
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_ang expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    call get_cell_from_arg(args, cell_p)
    ierror = ndarray_create(ang, cell_p%p%ang)
    ierror = dict_create(retval)
    ierror = retval%setitem("ang", ang)
    
    r = retval%get_c_ptr()
    
    call args%destroy

  end function crystal_metrics_get_ang

  function crystal_metrics_get_lcell(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Crystal_Cell_type_p) :: cell_p
  
    type(ndarray) :: lcell
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_lcell expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    call get_cell_from_arg(args, cell_p)
    
    ierror = ndarray_create(lcell, cell_p%p%lcell)
    
    ierror = dict_create(retval)
    
    ierror = retval%setitem("lcell", lcell)
    
    r = retval%get_c_ptr()
    
    call args%destroy

  end function crystal_metrics_get_lcell
  
  function crystal_metrics_get_lang(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Crystal_Cell_type_p) :: cell_p
  
    type(ndarray) :: lang
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_lang expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    call get_cell_from_arg(args, cell_p)
    
    ierror = ndarray_create(lang, cell_p%p%lang)
    
    ierror = dict_create(retval)
    
    ierror = retval%setitem("lang", lang)
    
    r = retval%get_c_ptr()
    
    call args%destroy

  end function crystal_metrics_get_lang
  
  function crystal_metrics_get_cell_std(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Crystal_Cell_type_p) :: cell_p
  
    type(ndarray) :: cell_std
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_cell_std expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    call get_cell_from_arg(args, cell_p)
    
    ierror = ndarray_create(cell_std, cell_p%p%cell_std)
    
    ierror = dict_create(retval)
    
    ierror = retval%setitem("cell_std", cell_std)
    
    r = retval%get_c_ptr()
    
    call args%destroy

  end function crystal_metrics_get_cell_std
  
  function crystal_metrics_get_ang_std(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Crystal_Cell_type_p) :: cell_p
  
    type(ndarray) :: ang_std
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_ang_std expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    call get_cell_from_arg(args, cell_p)
    
    ierror = ndarray_create(ang_std, cell_p%p%ang_std)
    
    ierror = dict_create(retval)
    
    ierror = retval%setitem("ang_std", ang_std)
    
    r = retval%get_c_ptr()
    
    call args%destroy

  end function crystal_metrics_get_ang_std
  
  function crystal_metrics_get_rcell(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Crystal_Cell_type_p) :: cell_p
  
    type(ndarray) :: rcell
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_rcell expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    call get_cell_from_arg(args, cell_p)
    
    ierror = ndarray_create(rcell, cell_p%p%rcell)
    
    ierror = dict_create(retval)
    
    ierror = retval%setitem("rcell", rcell)
    
    r = retval%get_c_ptr()

    call args%destroy
    
  end function crystal_metrics_get_rcell
  
  function crystal_metrics_get_rang(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Crystal_Cell_type_p) :: cell_p
  
    type(ndarray) :: rang
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_rang expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    call get_cell_from_arg(args, cell_p)
    
    ierror = ndarray_create(rang, cell_p%p%rang)
    
    ierror = dict_create(retval)
    
    ierror = retval%setitem("rang", rang)
    
    r = retval%get_c_ptr()
    
    call args%destroy

  end function crystal_metrics_get_rang
  
  function crystal_metrics_get_GD(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Crystal_Cell_type_p) :: cell_p
  
    type(ndarray) :: GD
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_GD expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    call get_cell_from_arg(args, cell_p)
    
    ierror = ndarray_create(GD, cell_p%p%GD)
    
    ierror = dict_create(retval)
    
    ierror = retval%setitem("GD", GD)
    
    r = retval%get_c_ptr()
    
    call args%destroy

  end function crystal_metrics_get_GD
  
  function crystal_metrics_get_GR(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Crystal_Cell_type_p) :: cell_p
  
    type(ndarray) :: GR
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_GR expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    call get_cell_from_arg(args, cell_p)
    
    ierror = ndarray_create(GR, cell_p%p%GR)
    
    ierror = dict_create(retval)
    
    ierror = retval%setitem("GR", GR)
    
    r = retval%get_c_ptr()
    
    call args%destroy

  end function crystal_metrics_get_GR
  
  function crystal_metrics_get_Cr_Orth_Cel(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Crystal_Cell_type_p) :: cell_p
  
    type(ndarray) :: Cr_Orth_Cel
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_Cr_Orth_Cel expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    call get_cell_from_arg(args, cell_p)
    
    ierror = ndarray_create(Cr_Orth_Cel, cell_p%p%Cr_Orth_Cel)
    
    ierror = dict_create(retval)
    
    ierror = retval%setitem("Cr_Orth_Cel", Cr_Orth_Cel)
    
    r = retval%get_c_ptr()
    
    call args%destroy

  end function crystal_metrics_get_Cr_Orth_Cel
  
  function crystal_metrics_get_Orth_Cr_Cel(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Crystal_Cell_type_p) :: cell_p
  
    type(ndarray) :: Orth_Cr_Cel
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_Orth_Cr_Cel expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    call get_cell_from_arg(args, cell_p)
    
    ierror = ndarray_create(Orth_Cr_Cel, cell_p%p%Orth_Cr_Cel)
    
    ierror = dict_create(retval)
    
    ierror = retval%setitem("Orth_Cr_Cel", Orth_Cr_Cel)
    
    r = retval%get_c_ptr()
    
    call args%destroy

  end function crystal_metrics_get_Orth_Cr_Cel
  
  function crystal_metrics_get_BL_M(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Crystal_Cell_type_p) :: cell_p
  
    type(ndarray) :: BL_M
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_BL_M expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    call get_cell_from_arg(args, cell_p)
    
    ierror = ndarray_create(BL_M, cell_p%p%BL_M)
    
    ierror = dict_create(retval)
    
    ierror = retval%setitem("BL_M", BL_M)
    
    r = retval%get_c_ptr()
    
    call args%destroy

  end function crystal_metrics_get_BL_M
  
  function crystal_metrics_get_BL_Minv(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Crystal_Cell_type_p) :: cell_p
  
    type(ndarray) :: BL_Minv
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_BL_Minv expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    call get_cell_from_arg(args, cell_p)
    
    ierror = ndarray_create(BL_Minv, cell_p%p%BL_Minv)
    
    ierror = dict_create(retval)
    
    ierror = retval%setitem("BL_Minv", BL_Minv)
    
    r = retval%get_c_ptr()
    
    call args%destroy

  end function crystal_metrics_get_BL_Minv
  
  function crystal_metrics_get_cellvol(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Crystal_Cell_type_p) :: cell_p
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_cellvol expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    call get_cell_from_arg(args, cell_p)
    
    ierror = dict_create(retval)
    
    ierror = retval%setitem("cellvol", cell_p%p%cellvol)
    
    r = retval%get_c_ptr()
    
    call args%destroy

  end function crystal_metrics_get_cellvol
  
  function crystal_metrics_get_rcellvol(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Crystal_Cell_type_p) :: cell_p
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_rcellvol expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    call get_cell_from_arg(args, cell_p)
    
    ierror = dict_create(retval)
    
    ierror = retval%setitem("rcellvol", cell_p%p%rcellvol)
    
    r = retval%get_c_ptr()

    call args%destroy
    
  end function crystal_metrics_get_rcellvol
  
  function crystal_metrics_get_stdvol(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Crystal_Cell_type_p) :: cell_p
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_stdvol expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    call get_cell_from_arg(args, cell_p)
    
    ierror = dict_create(retval)
    
    ierror = retval%setitem("stdvol", cell_p%p%stdvol)
    
    r = retval%get_c_ptr()
    
    call args%destroy

  end function crystal_metrics_get_stdvol
  
  function crystal_metrics_get_CartType(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Crystal_Cell_type_p) :: cell_p
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_CartType expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    call get_cell_from_arg(args, cell_p)
    
    ierror = dict_create(retval)
    
    ierror = retval%setitem("CartType", trim(cell_p%p%CartType))
    
    r = retval%get_c_ptr()
    
    call args%destroy

  end function crystal_metrics_get_CartType


  
  function crystal_metrics_cart_vector(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror

    type(Crystal_Cell_type_p) :: cell_p

    type(object)                         :: vec_obj
    type(ndarray)                        :: vec_nd
    real(kind=cp), dimension(:), pointer :: vec_p

    real(kind=cp), dimension(3)          :: cart_vec
    type(ndarray)                        :: cart_vec_nd

    character(len=:), allocatable :: code
    type(object)     :: code_obj

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 3) then
       call raise_exception(TypeError, "k_to_cart_vector expects exactly 3 arguments")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    ierror = args%getitem(code_obj, 0)        !code
    ierror = cast_nonstrict(code, code_obj)

    ierror = args%getitem(vec_obj, 1)        !vec -> kx,ky,kz
    ierror = cast(vec_nd, vec_obj)
    ierror = vec_nd%get_data(vec_p)

    call get_cell_from_arg(args, cell_p, 2)

    cart_vec = Cart_Vector(code, vec_p, cell_p%p)

    ierror = ndarray_create(cart_vec_nd, cart_vec)
    ierror = dict_create(retval)
    ierror = retval%setitem("cart_vec", cart_vec_nd)

    r = retval%get_c_ptr()

    call args%destroy
    call vec_obj%destroy
    call code_obj%destroy

  end function crystal_metrics_cart_vector

  function crystal_metrics_cart_u_vector(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror

    type(Crystal_Cell_type_p) :: cell_p

    type(object)                         :: vec_obj
    type(ndarray)                        :: vec_nd
    real(kind=cp), dimension(:), pointer :: vec_p

    real(kind=cp), dimension(3)          :: cart_vec
    type(ndarray)                        :: cart_vec_nd

    character(len=:), allocatable :: code
    type(object)     :: code_obj

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 3) then
       call raise_exception(TypeError, "k_to_cart_vector expects exactly 3 arguments")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    ierror = args%getitem(code_obj, 0)        !code
    ierror = cast_nonstrict(code, code_obj)

    ierror = args%getitem(vec_obj, 1)        !vec -> kx,ky,kz
    ierror = cast(vec_nd, vec_obj)
    ierror = vec_nd%get_data(vec_p)

    call get_cell_from_arg(args, cell_p, 2)

    cart_vec = Cart_U_Vector(code, vec_p, cell_p%p)

    ierror = ndarray_create(cart_vec_nd, cart_vec)
    ierror = dict_create(retval)
    ierror = retval%setitem("cart_vec", cart_vec_nd)

    r = retval%get_c_ptr()

    call args%destroy
    call vec_obj%destroy
    call code_obj%destroy

  end function crystal_metrics_cart_u_vector



  function crystal_metrics_get_betas_from_biso(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror

    type(Crystal_Cell_type_p) :: cell_p

    type(object)                         :: biso_obj
    real(kind=cp)                        :: biso

    real(kind=cp), dimension(6)          :: betas
    type(ndarray)                        :: betas_nd

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "betas_grom_biso expects exactly 2 arguments")
       call args%destroy
       return
    endif

    ! Doing boring stuff
    ierror = args%getitem(biso_obj, 0)        !code
    ierror = cast_nonstrict(biso, biso_obj)

    call get_cell_from_arg(args, cell_p, 1)

    betas = Get_Betas_from_Biso(Biso, cell_p%p)

    ierror = ndarray_create(betas_nd, betas)
    ierror = dict_create(retval)
    ierror = retval%setitem("betas", betas_nd)

    r = retval%get_c_ptr()

    call args%destroy
    call biso_obj%destroy
  
  end function crystal_metrics_get_betas_from_biso

end module API_Crystal_Metrics
