! **************************************************************************
!
! CrysFML API
!
! @file      Src/Extensions/API_Structure_Factors.f90
! @brief     CFML Structure Factors Fortran binding
!
! @homepage  https://code.ill.fr/scientific-software/crysfml
! @license   GNU LGPL (see LICENSE)
! @copyright Institut Laue Langevin 2020-now
! @authors   Scientific Computing Group at ILL (see AUTHORS), based on Elias Rabel work for Forpy
!
! **************************************************************************

module API_Structure_Factors

  use forpy_mod
  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env

  use CFML_GlobalDeps,                 only: Cp
  use CFML_Structure_Factors,          only: &
       Structure_Factors, &
       Write_Structure_Factors, &
       Set_Fixed_Tables

  use API_Crystallographic_Symmetry, only: &
       Space_Group_Type_p, &
       get_space_group_type_from_arg

  use API_Atom_TypeDef, only: &
       Atom_list_type_p, &
       get_atom_list_type_from_arg

  use API_Reflections_Utilities, only: &
      Reflection_List_type_p, &
      get_reflection_list_from_arg

  use API_IO_Formats, only: &
       Job_info_type_p, &
       get_job_info_type_from_arg

  implicit none

contains

!!$  function structure_factors_structure_factors(self_ptr, args_ptr) result(r) bind(c)
!!$
!!$    type(c_ptr), value :: self_ptr
!!$    type(c_ptr), value :: args_ptr
!!$    type(c_ptr)        :: r
!!$    type(tuple)        :: args
!!$    type(dict)         :: retval
!!$
!!$    integer            :: num_args
!!$    integer            :: ierror
!!$    integer            :: ii
!!$
!!$    type(list)   :: index_obj
!!$    type(object) :: arg_obj
!!$
!!$    type(Atom_list_type_p)          :: atom_list_p
!!$    type(Space_Group_type_p)        :: spg_p
!!$    type(Reflection_List_type_p)    :: reflection_list_p
!!$
!!$    character(len=3)       :: mode
!!$    character(len=16)      :: pattern
!!$    real(kind=cp)          :: lambda
!!$    
!!$    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
!!$    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
!!$    call unsafe_cast_from_c_ptr(args, args_ptr)
!!$    ! Check if the arguments are OK
!!$    ierror = args%len(num_args)
!!$    
!!$    if (num_args /= 3) then
!!$       call raise_exception(TypeError, "structure_factors_structure_factors expects exactly 3 arguments")
!!$       !@atom_list @space_group.as_fortran_object(), @job_info, @reflection_list
!!$       call args%destroy
!!$       return
!!$    endif
!!$
!!$    call get_atom_list_type_from_arg(args, atom_list_p, 0)
!!$
!!$    call get_space_group_type_from_arg(args, spg_p, 1)
!!$
!!$    call get_reflection_list_from_arg(args, reflection_list_p, 2)
!!$    
!!$    call Structure_Factors(atom_list_p%p, spg_p%p, reflection_list_p%p)
!!$
!!$    ierror = dict_create(retval)
!!$    r = retval%get_c_ptr()
!!$
!!$  end function structure_factors_structure_factors

  function structure_factors_structure_factors(self_ptr, args_ptr) result(r) bind(c)

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

    type(Atom_list_type_p)          :: atom_list_p
    type(Space_Group_type_p)        :: spg_p
    type(Reflection_List_type_p)    :: reflection_list_p
    type(job_info_type_p)           :: job_p

    character(len=3)       :: mode
    character(len=16)      :: pattern
    real(kind=cp)          :: lambda
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    
    if (num_args /= 4) then
       call raise_exception(TypeError, "structure_factors_structure_factors expects exactly 4 arguments")
       !@atom_list @space_group, @reflection_list,  @job_info
       call args%destroy
       return
    endif

    call get_atom_list_type_from_arg(args, atom_list_p, 0)

    call get_space_group_type_from_arg(args, spg_p, 1)

    call get_reflection_list_from_arg(args, reflection_list_p, 2)

    call get_job_info_type_from_arg(args, job_p, 3)

    write(*,*) job_p%p%patt_typ(1)

    select case (job_p%p%patt_typ(1))
    case ("XRAY_2THE", "XRAY_SXTAL", "XRAY_ENER")
       mode = "XRA"
       !write(*,*) "X-Ray calculation"
       lambda = job_p%p%lambda(1)%mina
       call Structure_Factors(atom_list_p%p, spg_p%p, reflection_list_p%p, mode, lambda)
       
    case("NEUT_2THE", "NEUT_SXTAL", "NEUT_TOF" )
       mode = "NUC"
       !write(*,*) "Neutron calculation"
       call Structure_Factors(atom_list_p%p, spg_p%p, reflection_list_p%p, mode) 
       
    case default
       write(*,*) 'Default calculation'
       call Structure_Factors(atom_list_p%p, spg_p%p, reflection_list_p%p)
    end select

       

       !!----
    !!---- Subroutine Structure_Factors(Atm,Grp,Reflex,Mode,lambda)
    !!----    type(atom_list_type),               intent(in)     :: Atm    !List of atoms
    !!----    type(space_group_type),             intent(in)     :: Grp    !Space group
    !!----    type(reflection_list_type),         intent(in out) :: Reflex !It is completed on output
    !!----    character(len=*), optional,         intent(in)     :: Mode   !"NUC","ELE" for neutrons, electrons else: XRays
    !!----    real(kind=cp), optional,            intent(in)     :: lambda !Needed for Xrays
    !!----
    !!----    Calculate the Structure Factors from a list of Atoms
    !!----    and a set of reflections. A call to Init_Structure_Factors
    !!----    is a pre-requisite for using this subroutine. In any case
    !!----    the subroutine calls Init_Structure_Factors if SF_initialized=.false.
    !!----
    !!---- Update: February - 2005
    !!
    !call Structure_Factors(atom_list_p%p, spg_p%p, reflection_list_p%p) !, mode, lambda)

    ierror = dict_create(retval)
    r = retval%get_c_ptr()
   end function structure_factors_structure_factors

  function create_table_af0_xray_fun(self_ptr, args_ptr) result(r) bind(c)

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
    type(list) :: atom_list
    type(list) :: h_list, k_list, l_list  ! h, k, l 用のリストを作成

    type(Atom_list_type_p)          :: atom_list_p
    type(Space_Group_type_p)        :: spg_p
    type(Reflection_List_type_p)    :: reflection_list_p
    type(job_info_type_p)           :: job_p

    character(len=16)      :: mode
    real(kind=cp)          :: lambda
    integer                :: lun

    character(len=32) :: af0_str
    integer :: i, j

    ! 一時変数の宣言
    real(kind=cp), dimension(:), allocatable :: af0_slice
    character(len=32) :: h_str

     ! af0 配列を宣言
    real(kind=cp), dimension(:,:), allocatable :: af0
    integer :: total_size, idx
    real(kind=cp), allocatable :: af0_1d(:)
    type(list) :: af0_list  ! af0用のリストを作成

    r = C_NULL_PTR   ! エラー時の返り値をNULLポインタに初期化

    ! unsafe_cast_from_c_ptrを使ってc_ptrからFortranのデータ型にキャスト
    call unsafe_cast_from_c_ptr(args, args_ptr)

    ! 引数の数をチェック
    ierror = args%len(num_args)

    if (num_args /= 4 .and. num_args /= 5) then
       call raise_exception(TypeError, "create_table_af0_xray expects 4 or 5 arguments")
       ! 引数のリストを破棄して終了
       call args%destroy
       return
    endif


    ! 引数を取得
    call get_atom_list_type_from_arg(args, atom_list_p, 0)
    call get_space_group_type_from_arg(args, spg_p, 1)
    call get_reflection_list_from_arg(args, reflection_list_p, 2)
    call get_job_info_type_from_arg(args, job_p, 3)

    print *, "今までの処理"

    ! Create_Table_AF0_Xrayサブルーチンを呼び出し
    ! call Create_Table_AF0_Xray(reflection_list_p%p, atom_list_p%p)
    mode = "XRA"
    !write(*,*) "X-Ray calculation"
    lambda = job_p%p%lambda(1)%mina
    call Set_Fixed_Tables(reflection_list_p%p, atom_list_p%p, spg_p%p, mode, lambda, af0=af0)
    

    ! ---- 辞書型で結果を返す ----!
    ! 辞書作成
    ierror = dict_create(retval)
    
    !call dict_add_value(retval, "test", "test")
    ! atom_listの情報をリストに格納
   ierror = list_create(atom_list)  ! atom用のリストを作成
   do ii = 1, atom_list_p%p%natoms
      ierror = atom_list%append(atom_list_p%p%atom(ii)%chemsymb)  ! リストに追加
   end do
   ierror = retval%setitem("atom", atom_list)  ! 最終的に辞書にセット

   ! 各リストを作成
   ierror = list_create(h_list)
   ierror = list_create(k_list)
   ierror = list_create(l_list)
   ! h, k, l の値をリストに追加
   do ii = 1, reflection_list_p%p%nref
      ! h(1), h(2), h(3) をそれぞれリストに追加
      ierror = h_list%append(reflection_list_p%p%ref(ii)%h(1))  ! hリストに追加
      ierror = k_list%append(reflection_list_p%p%ref(ii)%h(2))  ! kリストに追加
      ierror = l_list%append(reflection_list_p%p%ref(ii)%h(3))  ! lリストに追加
   end do

   ! 最終的に辞書にセット
   ierror = retval%setitem("h", h_list)
   ierror = retval%setitem("k", k_list)
   ierror = retval%setitem("l", l_list)

   ! af0の1次元配列化
   total_size = size(af0, 1) * size(af0, 2)  ! 2次元配列の全要素数
   allocate(af0_1d(total_size))  ! 1次元配列を作成

   ! 2次元配列を1次元配列に変換
   idx = 1
   do i = 1, size(af0, 1)
      do j = 1, size(af0, 2)
         af0_1d(idx) = af0(i, j)
         idx = idx + 1
      end do
   end do

   ! af0用のリストを作成
   ierror = list_create(af0_list)

   ! af0_1dの各要素をリストに追加
   do idx = 1, total_size
      ierror = af0_list%append(af0_1d(idx))  ! 1次元配列の要素をリストに追加
   end do
   ! af0_listを辞書にセット
   ierror = retval%setitem("af0", af0_list)
    

    ! 結果をPythonに返す
    r = retval%get_c_ptr()

   end function create_table_af0_xray_fun

  function structure_factors_write_structure_factors(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Reflection_List_type_p)    :: reflection_list_p

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "write_structure_factors expects exactly 1 argument")
       call args%destroy
       return
    endif

    !
    call get_reflection_list_from_arg(args, reflection_list_p)

    !
    call Write_Structure_Factors(6, reflection_list_p%p)

    !
    ierror = dict_create(retval)
    r = retval%get_c_ptr()

  end function structure_factors_write_structure_factors

end module API_Structure_Factors
