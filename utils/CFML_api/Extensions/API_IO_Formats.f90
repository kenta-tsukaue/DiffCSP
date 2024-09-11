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

module API_IO_Formats
  
  use forpy_mod
  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env 

  use CFML_GlobalDeps,                only: cp, sp, pi
  use CFML_IO_Formats,                only: &
       Job_Info_type, &
       Get_Job_Info, &
       Readn_set_Xtal_structure
  Use CFML_Math_General,              only: Sind
  
  use API_Atom_TypeDef,               only: Atom_List_Type_p  
  use API_Crystallographic_Symmetry,  only: Space_Group_Type_p
  use API_Crystal_Metrics,            only: Crystal_Cell_Type_p

  implicit none

  type Job_Info_type_p
     type(Job_Info_type), pointer :: p
  end type Job_Info_type_p

contains
  
  !-------------------------------------------------------------------------
  ! Implementation of our Python methods
  !-------------------------------------------------------------------------
  ! @brief 
  function IO_formats_readn_set_xtal_structure(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr)        :: r
    type(tuple)        :: args 
    type(dict)         :: retval
    integer            :: num_args
    integer            :: ierror
    integer            :: ii

    type(object)                    :: filename_obj, mode_obj
    character(len=:), allocatable   :: filename, mode

    type(Crystal_Cell_type_p)       :: cell_p
    type(Space_Group_type_p)        :: spg_p
    type(Atom_list_type_p)          :: a_p
    type(Job_info_type_p)           :: job_p

    integer                         :: cell_p12(12), spg_p12(12), a_p12(12), job_p12(12)
    type(list)                      :: cell_obj, spg_obj, a_obj, job_obj

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)

    if (num_args /= 2) then
       call raise_exception(TypeError, "readn_set_xtal_structure expects exactly 2 arguments: filename, mode")
       call args%destroy
       return
    endif

    ierror = args%getitem(filename_obj, 0)
    ierror = cast(filename, filename_obj)

    ierror = args%getitem(mode_obj, 1)
    ierror = cast_nonstrict(mode, mode_obj)

    allocate(cell_p%p, spg_p%p, a_p%p, job_p%p)
    

    if (mode == 'CIF') then
       call Readn_set_Xtal_structure(trim(filename),cell_p%p,spg_p%p,a_p%p,Mode='CIF')

       call Init_Job_info(job_p%p)
       
    else
       call Readn_set_Xtal_structure(trim(filename),cell_p%p,spg_p%p,a_p%p,Mode=mode,Job_Info=job_p%p)
    end if

    if (job_p%p%num_patterns /= 1) then
       write(*,*) 'Warning: only one pattern is possible, only the first one will be considered'
    endif

    if (job_p%p%lambda(1)%mina /= job_p%p%lambda(1)%maxb) then
       write(*,*) 'Warning: only monochromatic calculation is implemented, lambda is set to lambda_min'
       job_p%p%lambda(1)%maxb = job_p%p%lambda(1)%mina
    endif

       
    cell_p12 = transfer(cell_p, cell_p12)
    ierror = list_create(cell_obj)
    do ii=1,12
       ierror = cell_obj%append(cell_p12(ii))
    end do

    spg_p12   = transfer(spg_p, spg_p12)
    ierror = list_create(spg_obj)
    do ii=1,12
       ierror = spg_obj%append(spg_p12(ii))
    end do

    a_p12    = transfer(a_p, a_p12)
    ierror = list_create(a_obj)
    do ii=1,12
       ierror = a_obj%append(a_p12(ii))
    end do

    job_p12    = transfer(job_p, job_p12)
    ierror = list_create(job_obj)
    do ii=1,12
       ierror = job_obj%append(job_p12(ii))
    end do

    ierror = dict_create(retval)
    ierror = retval%setitem("Cell", cell_obj)
    ierror = retval%setitem("SpG",  spg_obj)
    ierror = retval%setitem("A",    a_obj) 
    ierror = retval%setitem("JobInfo", job_obj)

    r = retval%get_c_ptr()
    call args%destroy

  end function IO_formats_readn_set_xtal_structure

  subroutine Init_Job_info(job_info)

    type(Job_Info_type) :: job_info

    integer                           :: n_pat, nphas
    real(kind=sp)                     :: a1,a2,a3,a4,a5

    n_pat=1
    nphas=1
    
    Job_info%title=" General Job: CrysFML"
    Job_info%Num_Patterns=1
    Job_info%Num_Patterns = 1
    Job_info%Num_Phases= 1
    Job_info%Num_Cmd= 0

    if (allocated(Job_Info%Patt_typ)) deallocate(Job_Info%Patt_typ)
    allocate(Job_Info%Patt_typ(n_pat))

    if (allocated(Job_Info%Phas_nam)) deallocate(Job_Info%Phas_nam)
    allocate(Job_Info%Phas_nam(nphas))

    if (allocated(Job_Info%range_stl)) deallocate(Job_Info%range_stl)
    allocate(Job_Info%range_stl(n_pat))

    if (allocated(Job_Info%range_q)) deallocate(Job_Info%range_q)
    allocate(Job_Info%range_q(n_pat))

    if (allocated(Job_Info%range_d)) deallocate(Job_Info%range_d)
    allocate(Job_Info%range_d(n_pat))

    if (allocated(Job_Info%range_2theta)) deallocate(Job_Info%range_2theta)
    allocate(Job_Info%range_2theta(n_pat))

    if (allocated(Job_Info%range_energy)) deallocate(Job_Info%range_energy)
    allocate(Job_Info%range_energy(n_pat))

    if (allocated(Job_Info%range_tof)) deallocate(Job_Info%range_tof)
    allocate(Job_Info%range_tof(n_pat))

    if (allocated(Job_Info%lambda)) deallocate(Job_Info%lambda)
    allocate(Job_Info%lambda(n_pat))

    if (allocated(Job_Info%ratio)) deallocate(Job_Info%ratio)
    allocate(Job_Info%ratio(n_pat))

    if (allocated(Job_Info%dtt1)) deallocate(Job_Info%dtt1)
    allocate(Job_Info%dtt1(n_pat))

    if (allocated(Job_Info%dtt2)) deallocate(Job_Info%dtt2)
    allocate(Job_Info%dtt2(n_pat))

    Job_Info%Patt_typ    =" "
    Job_Info%Phas_nam    =" "
    Job_Info%range_stl%mina=0.0
    Job_Info%range_stl%maxb=0.0
    Job_Info%range_q%mina=0.0
    Job_Info%range_q%maxb=0.0
    Job_Info%range_d%mina=0.0
    Job_Info%range_d%maxb=0.0
    Job_Info%range_2theta%mina=0.0
    Job_Info%range_2theta%maxb=0.0
    Job_Info%range_Energy%mina=0.0
    Job_Info%range_Energy%maxb=0.0
    Job_Info%range_tof%mina=0.0
    Job_Info%range_tof%maxb=0.0
    Job_Info%Lambda%mina=0.0
    Job_Info%Lambda%maxb=0.0
    Job_Info%ratio = 0.0
    Job_Info%dtt1 = 0.0
    Job_Info%dtt2 = 0.0
    
    
    a1=1.5405
    a2=a1
    a3=0.0
    a4=0.0
    a5=120.0
    Job_Info%Patt_typ(n_pat)="XRAY_2THE"
    Job_Info%Lambda(n_pat)%mina=a1
    Job_Info%Lambda(n_pat)%maxb=a2
    Job_Info%ratio(n_pat)=a3
    Job_Info%range_2theta(n_pat)%mina=a4
    Job_Info%range_2theta(n_pat)%maxb=a5
    a4=sind(0.5*a4)/a1
    a5=sind(0.5*a5)/a2
    Job_Info%range_stl(n_pat)%mina=a4
    Job_Info%range_stl(n_pat)%maxb=a5
    Job_Info%range_q(n_pat)%mina=a4*4.0*pi
    Job_Info%range_q(n_pat)%maxb=a5*4.0*pi
    Job_Info%range_d(n_pat)%mina=0.5/a5
    Job_Info%range_d(n_pat)%maxb=0.5/a4

    Job_Info%Phas_nam(1)= Job_info%title

    Job_info%U=0.02
    Job_info%V=-0.02
    Job_info%W=0.12
    Job_info%X=0.0015
    Job_info%Y= 0.0465
    Job_info%theta_Step= 0.05

  end subroutine Init_Job_info

  ! @brief Create an atom list from an array of CIF lines
  function IO_Formats_jobinfo_from_CIF_string_array(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r

    type(tuple) :: args
    type(dict)  :: retval
    integer     :: num_args
    integer     :: ierror

    integer     :: max_line_length
    integer     :: ii
    type(list)  :: job_obj
    type(object):: item_obj
    character(len=:), allocatable :: item

    integer                              :: nline, n_ini
    type(Job_Info_type_p)                :: job_p
    integer                              :: job_p12(12)

    character(len=:), dimension(:), allocatable :: stringarray
    type(object)                            :: stringarray_obj
    type(list)                           :: stringarray_list
    
    r = C_NULL_PTR      
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ierror = args%len(num_args)
    if (num_args /=1 ) then
       call raise_exception(TypeError, "jobinfo_from_CIF_string_array expects exactly 1 argument")
       call args%destroy
       return
    endif

    ierror = args%getitem(stringarray_obj, 0)
    ierror = cast(stringarray_list, stringarray_obj)
    ierror = stringarray_list%len(nline)

    ! Compute max line length
    max_line_length = 0
    do ii=1,nline
        ierror = stringarray_list%getitem(item_obj, ii-1)
        ierror = cast(item, item_obj)
        if (len(item) > max_line_length) then
            max_line_length = len(item)
        endif
        call item_obj%destroy
    enddo
    allocate(character(max_line_length) :: stringarray(nline))

    ! Fill stringarray
    do ii=1,nline
        ierror = stringarray_list%getitem(item_obj, ii-1)
        ierror = cast(item, item_obj)
        stringarray(ii) = item
        call item_obj%destroy
    enddo

    ! Create atom list from stringarray
    allocate(job_p%p)
    n_ini = 1
    call Get_Job_Info(stringarray, n_ini, nline+1, job_p%p)
    !
    job_p12 = transfer(job_p,job_p12)
    ierror = list_create(job_obj)
    do ii=1,12
       ierror = job_obj%append(job_p12(ii))
    end do

    !
    ierror = dict_create(retval)
    ierror = retval%setitem("JobInfo", job_obj)
    r = retval%get_c_ptr()

    !
    call args%destroy
    call job_obj%destroy
    deallocate(stringarray)
    call stringarray_obj%destroy
    call stringarray_list%destroy

  end function IO_Formats_jobinfo_from_CIF_string_array

  subroutine get_job_info_type_from_arg(args, job_info_type_pointer, indx)
    
    type(tuple) :: args
    type(Job_info_type_p), intent(out) :: job_info_type_pointer
    integer, optional    :: indx

    type(object) :: arg_obj
    type(list) :: arg_list
    integer :: job_info_type_p12(12)

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
       ierror = cast(job_info_type_p12(ii), t)
       call t%destroy
    enddo
    job_info_type_pointer = transfer(job_info_type_p12, job_info_type_pointer)
    call arg_obj%destroy
    call arg_list%destroy
    
  end subroutine get_job_info_type_from_arg

  
  function IO_Formats_del_jobinfo(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
  
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(job_info_type_p) :: job_p

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "del_jobinfo expects exactly 1 argument")
       call args%destroy
       return
    endif

    !
    call get_job_info_type_from_arg(args, job_p)

    if(allocated(job_p%p%Patt_typ))      deallocate(job_p%p%Patt_typ)
    if(allocated(job_p%p%Phas_nam))      deallocate(job_p%p%Phas_nam)
    if(allocated(job_p%p%cmd))           deallocate(job_p%p%cmd)

    if(allocated(job_p%p%range_stl))     deallocate(job_p%p%range_stl)   
    if(allocated(job_p%p%range_q))       deallocate(job_p%p%range_q)    
    if(allocated(job_p%p%range_d))       deallocate(job_p%p%range_d)    
    if(allocated(job_p%p%range_2theta))  deallocate(job_p%p%range_2theta)
    if(allocated(job_p%p%range_Energy))  deallocate(job_p%p%range_Energy)
    if(allocated(job_p%p%range_tof))     deallocate(job_p%p%range_tof)
    if(allocated(job_p%p%Lambda))        deallocate(job_p%p%Lambda)  
    if(allocated(job_p%p%ratio))         deallocate(job_p%p%ratio)     
    if(allocated(job_p%p%dtt1))          deallocate(job_p%p%dtt1)
    if(allocated(job_p%p%dtt2))          deallocate(job_p%p%dtt2)

    deallocate(job_p%p)
    !
    ierror = dict_create(retval)
    r = retval%get_c_ptr()

    call args%destroy
    
  end function IO_Formats_del_jobinfo

  function IO_Formats_get_title(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_info_type_pointer
  
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
    call get_job_info_type_from_arg(args, job_info_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("title", trim(job_info_type_pointer%p%title))
    
    r = retval%get_c_ptr()
    call args%destroy
     
  end function IO_Formats_get_title
  
  
  function IO_Formats_get_num_phases(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_info_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_num_phases expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_info_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("num_phases", job_info_type_pointer%p%num_phases)
    r = retval%get_c_ptr()
    call args%destroy
     
  end function IO_Formats_get_num_phases
  
  
  function IO_Formats_get_num_patterns(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_info_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_num_patterns expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_info_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("num_patterns", job_info_type_pointer%p%num_patterns)
    r = retval%get_c_ptr()
    call args%destroy
     
  end function IO_Formats_get_num_patterns
  
  
  function IO_Formats_get_num_cmd(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_info_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_num_cmd expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_info_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("num_cmd", job_info_type_pointer%p%num_cmd)
    r = retval%get_c_ptr()
    call args%destroy
     
  end function IO_Formats_get_num_cmd
  
  
  function IO_Formats_get_patt_typ(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_info_type_pointer

    type(object) :: indx_obj
    integer :: indx
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "get_patt_typ expects exactly 2 arguments")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_info_type_pointer)

    
    ierror = args%getitem(indx_obj, 1) !get the pattern number
    ierror = cast_nonstrict(indx, indx_obj)
    
    ierror = dict_create(retval)
    ierror = retval%setitem("patt_typ", job_info_type_pointer%p%patt_typ(indx))
    
    r = retval%get_c_ptr()
    call args%destroy
    call indx_obj%destroy
     
  end function IO_Formats_get_patt_typ
  
  
  function IO_Formats_get_phas_nam(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_info_type_pointer

    type(object) :: indx_obj
    integer :: indx
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "get_phas_nam expects exactly 2 arguments")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_info_type_pointer)

    ierror = args%getitem(indx_obj, 1) !get the phase number
    ierror = cast_nonstrict(indx, indx_obj)
    
    ierror = dict_create(retval)
    ierror = retval%setitem("phas_nam", job_info_type_pointer%p%phas_nam(indx))
    
    r = retval%get_c_ptr()
    call args%destroy
    call indx_obj%destroy
     
  end function IO_Formats_get_phas_nam
  
  
  function IO_Formats_get_cmd(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_info_type_pointer

    type(object) :: indx_obj
    integer :: indx
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "get_cmd expects exactly 2 arguments")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_info_type_pointer)

    ierror = args%getitem(indx_obj, 1) !get the phase number
    ierror = cast_nonstrict(indx, indx_obj)
    
    ierror = dict_create(retval)
    ierror = retval%setitem("cmd", job_info_type_pointer%p%cmd(indx))
    
    r = retval%get_c_ptr()
    call args%destroy
    call indx_obj%destroy
     
  end function IO_Formats_get_cmd
  
  
  function IO_Formats_get_range_stl(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_info_type_pointer

    type(object) :: indx_obj
    integer :: indx
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "get_range_stl expects exactly 2 arguments")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_info_type_pointer)

    ierror = args%getitem(indx_obj, 1) !get the phase number
    ierror = cast_nonstrict(indx, indx_obj)
    
    ierror = dict_create(retval)
    ierror = retval%setitem("min", job_info_type_pointer%p%range_stl(indx)%mina)
    ierror = retval%setitem("max", job_info_type_pointer%p%range_stl(indx)%maxb)
    
    r = retval%get_c_ptr()
    call args%destroy
    call indx_obj%destroy
     
  end function IO_Formats_get_range_stl
  
  
  function IO_Formats_get_range_q(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_info_type_pointer

    type(object) :: indx_obj
    integer :: indx
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "get_range_q expects exactly 2 arguments")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_info_type_pointer)

    ierror = args%getitem(indx_obj, 1) !get the phase number
    ierror = cast_nonstrict(indx, indx_obj)
    
    ierror = dict_create(retval)
    ierror = retval%setitem("min", job_info_type_pointer%p%range_q(indx)%mina)
    ierror = retval%setitem("max", job_info_type_pointer%p%range_q(indx)%maxb)
    
    r = retval%get_c_ptr()
    call args%destroy
    call indx_obj%destroy
     
  end function IO_Formats_get_range_q
  
  
  function IO_Formats_get_range_d(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_info_type_pointer
  
    type(object) :: indx_obj
    integer :: indx
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "get_range_d expects exactly 2 arguments")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_info_type_pointer)

    ierror = args%getitem(indx_obj, 1) !get the phase number
    ierror = cast_nonstrict(indx, indx_obj)
    
    ierror = dict_create(retval)
    ierror = retval%setitem("min", job_info_type_pointer%p%range_d(indx)%mina)
    ierror = retval%setitem("max", job_info_type_pointer%p%range_d(indx)%maxb)
    
    r = retval%get_c_ptr()
    call args%destroy
    call indx_obj%destroy
     
  end function IO_Formats_get_range_d
  
  
  function IO_Formats_get_range_2theta(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_info_type_pointer

    type(object) :: indx_obj
    integer :: indx
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "get_range_2theta expects exactly 2 arguments")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_info_type_pointer)
    
    ierror = args%getitem(indx_obj, 1) !get the phase number
    ierror = cast_nonstrict(indx, indx_obj)
    
    ierror = dict_create(retval)
    ierror = retval%setitem("min", job_info_type_pointer%p%range_2theta(indx)%mina)
    ierror = retval%setitem("max", job_info_type_pointer%p%range_2theta(indx)%maxb)
    
    r = retval%get_c_ptr()
    call args%destroy
    call indx_obj%destroy
    
     
  end function IO_Formats_get_range_2theta

 
  function IO_Formats_get_range_energy(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_info_type_pointer
  
    type(object) :: indx_obj
    integer :: indx
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "get_range_energy expects exactly 2 arguments")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_info_type_pointer)

    ierror = args%getitem(indx_obj, 1) !get the phase number
    ierror = cast_nonstrict(indx, indx_obj)
    
    ierror = dict_create(retval)
    ierror = retval%setitem("min", job_info_type_pointer%p%range_energy(indx)%mina)
    ierror = retval%setitem("max", job_info_type_pointer%p%range_energy(indx)%maxb)
    
    r = retval%get_c_ptr()
    call args%destroy
    call indx_obj%destroy  
     
  end function IO_Formats_get_range_energy
  
  
  function IO_Formats_get_lambda(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_info_type_pointer

    type(object) :: indx_obj
    integer :: indx
    
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "get_lambda expects exactly 2 arguments")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_info_type_pointer)
    
    ierror = args%getitem(indx_obj, 1) !get the phase number
    ierror = cast_nonstrict(indx, indx_obj)
    
    ierror = dict_create(retval)
    ierror = retval%setitem("lambda1", job_info_type_pointer%p%lambda(indx)%mina)
    ierror = retval%setitem("lambda2", job_info_type_pointer%p%lambda(indx)%maxb)
    
    r = retval%get_c_ptr()
    call args%destroy
    call indx_obj%destroy  

  end function IO_Formats_get_lambda
  
  
  function IO_Formats_get_ratio(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_info_type_pointer

    type(object) :: indx_obj
    integer :: indx
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "get_ratio expects exactly 2 argumenta")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_info_type_pointer)

    ierror = args%getitem(indx_obj, 1) !get the phase number
    ierror = cast_nonstrict(indx, indx_obj)
    
    ierror = dict_create(retval)
    ierror = retval%setitem("ratio", job_info_type_pointer%p%ratio(indx))
    r = retval%get_c_ptr()
    
    call args%destroy
    call indx_obj%destroy  
     
  end function IO_Formats_get_ratio
  
  
  function IO_Formats_get_dtt1(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_info_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_dtt1 expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_info_type_pointer)
    ierror = dict_create(retval)
    !ierror = retval%setitem("dtt1", job_info_type_pointer%p%dtt1)
    r = retval%get_c_ptr()
    call args%destroy
     
  end function IO_Formats_get_dtt1
  
  
  function IO_Formats_get_dtt2(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_info_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_dtt2 expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_info_type_pointer)
    ierror = dict_create(retval)
    !ierror = retval%setitem("dtt2", job_info_type_pointer%p%dtt2)
    r = retval%get_c_ptr()
    call args%destroy
     
  end function IO_Formats_get_dtt2


  function IO_Formats_get_U(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_info_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_U expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_info_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("U", job_info_type_pointer%p%U)
    r = retval%get_c_ptr()
    call args%destroy
     
  end function IO_Formats_get_U
  
  
  function IO_Formats_get_V(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_info_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_V expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_info_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("V", job_info_type_pointer%p%V)
    r = retval%get_c_ptr()
    call args%destroy
     
  end function IO_Formats_get_V
  
  
  function IO_Formats_get_W(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_info_type_pointer
  
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
    call get_job_info_type_from_arg(args, job_info_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("W", job_info_type_pointer%p%W)
    r = retval%get_c_ptr()
    call args%destroy
     
  end function IO_Formats_get_W
  
  
  function IO_Formats_get_X(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_info_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_X expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_info_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("X", job_info_type_pointer%p%X)
    r = retval%get_c_ptr()
    call args%destroy
     
  end function IO_Formats_get_X
  
  
  function IO_Formats_get_Y(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_info_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_Y expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_info_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("Y", job_info_type_pointer%p%Y)
    r = retval%get_c_ptr()
    call args%destroy
     
  end function IO_Formats_get_Y
  
  
  function IO_Formats_get_theta_step(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_info_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_theta_step expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_info_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("theta_step", job_info_type_pointer%p%theta_step)
    r = retval%get_c_ptr()
    call args%destroy
     
  end function IO_Formats_get_theta_step
  
  
  function IO_Formats_get_bkg(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_info_type_pointer
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 1) then
       call raise_exception(TypeError, "get_bkg expects exactly 1 argument")
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_info_type_pointer)
    ierror = dict_create(retval)
    ierror = retval%setitem("bkg", job_info_type_pointer%p%bkg)
    r = retval%get_c_ptr()
    call args%destroy
     
  end function IO_Formats_get_bkg

  !---------------------
  !Setters
  !---------------------
  function IO_Formats_set_patt_typ(self_ptr, args_ptr) result(r) bind(c)

    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_p

    type(object) :: val_obj, indx_obj
    character(len=:), allocatable :: val
    integer :: indx

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 3) then
       call raise_exception(TypeError, "set_pattern_types expects exactly 3 arguments")
       !self, new_value
       call args%destroy
       return
    endif

    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_p)

    
    ierror = args%getitem(val_obj, 1) !get the value
    ierror = cast_nonstrict(val, val_obj)

    ierror = args%getitem(indx_obj, 2) !get the indx
    ierror = cast_nonstrict(indx, indx_obj)

    
    select case (val)
    case ("XRAY_2THE", "XRAY_SXTAL", "XRAY_ENER","NEUT_2THE", "NEUT_SXTAL", "NEUT_TOF" )
       job_p%p%patt_typ(indx) = val
    case default
       write(*,*) val, " is not a valid pattern type"
    end select
       
    ierror = dict_create(retval)
    r = retval%get_c_ptr()

    call args%destroy
    call val_obj%destroy
    call indx_obj%destroy
    
    
  end function IO_Formats_set_patt_typ
  

    function IO_Formats_set_range_2theta(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_p

    type(object) :: indx_obj, mina_obj, maxb_obj
    integer :: indx

    real(kind=sp) :: a1,a2,a4,a5, mina, maxb

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args .le. 2) then
       call raise_exception(TypeError, "set_range_2theta expects 3 or more arguments")
       call args%destroy
       return
    endif

     ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_p)

    ierror = args%getitem(mina_obj, 1) !get the value of theta min
    ierror = cast_nonstrict(mina, mina_obj)

    ierror = args%getitem(maxb_obj, 2) !get the value of theta max
    ierror = cast_nonstrict(maxb, maxb_obj)

    ierror = args%getitem(indx_obj, 3) !get the phase number
    ierror = cast_nonstrict(indx, indx_obj)
    
    select case (job_p%p%patt_typ(indx))
    case("XRAY_2THE","NEUT_2THE","XRAY_SXTA","NEUT_SXTA")
       
       job_p%p%range_2theta(indx)%mina = mina
       job_p%p%range_2theta(indx)%maxb = maxb

       
       a1 = Job_p%p%Lambda(indx)%mina
       a2 = Job_p%p%Lambda(indx)%maxb
       
       a4=sind(0.5*mina)/a1
       a5=sind(0.5*maxb)/a2
       Job_p%p%range_stl(indx)%mina=a4
       Job_p%p%range_stl(indx)%maxb=a5
       Job_p%p%range_q(indx)%mina=a4*4.0*pi
       Job_p%p%range_q(indx)%maxb=a5*4.0*pi
       Job_p%p%range_d(indx)%mina=0.5/a5
       Job_p%p%range_d(indx)%maxb=0.5/a4
       
    case default
       write(*,*) "only possible to change theta in XRAY_2THE, NEUT_2THE, XRAY_SXTA, NEUT_SXTA"
    end select
         
    
    ierror = dict_create(retval)
    r = retval%get_c_ptr()

    call args%destroy
    call indx_obj%destroy
    call mina_obj%destroy
    call maxb_obj%destroy

  end function IO_Formats_set_range_2theta

  function IO_Formats_set_lambda(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_p

    type(object) :: indx_obj, mina_obj, maxb_obj
    integer :: indx

    real(kind=sp) :: mina, maxb

    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args .le. 2) then
       call raise_exception(TypeError, "set_range_lambda expects 3 or more arguments")
       call args%destroy
       return
    endif

     ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_p)

    ierror = args%getitem(mina_obj, 1) !get the value of lambda min
    ierror = cast_nonstrict(mina, mina_obj)

    ierror = args%getitem(maxb_obj, 2) !get the value of lambda max
    ierror = cast_nonstrict(maxb, maxb_obj)

    ierror = args%getitem(indx_obj, 3) !get the phase number
    ierror = cast_nonstrict(indx, indx_obj)
    
    select case (job_p%p%patt_typ(indx))
    case("XRAY_2THE","NEUT_2THE","XRAY_SXTA","NEUT_SXTA")
       
       job_p%p%lambda(indx)%mina = mina
       job_p%p%lambda(indx)%maxb = mina
       
    case default
       write(*,*) "only possible to change lambda in XRAY_2THE, NEUT_2THE, XRAY_SXTA, NEUT_SXTA"
    end select
         
    
    ierror = dict_create(retval)
    r = retval%get_c_ptr()

    call args%destroy
    call indx_obj%destroy
    call mina_obj%destroy
    call maxb_obj%destroy

  end function IO_Formats_set_lambda

  function IO_Formats_set_ratio(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_p

    type(object) :: val_obj, indx_obj
    real(kind=cp) :: val
    integer :: indx
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 3) then
       call raise_exception(TypeError, "set_lambda_ratio expects exactly 3 arguments")
       !self, new_value, indx
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_p)

    ierror = args%getitem(val_obj, 1) !get the value
    ierror = cast_nonstrict(val, val_obj)

    ierror = args%getitem(indx_obj, 2) !get the phase number
    ierror = cast_nonstrict(indx, indx_obj)

    job_p%p%ratio(indx) = val
    
    ierror = dict_create(retval)
    r = retval%get_c_ptr()

    call args%destroy
    call val_obj%destroy
    call indx_obj%destroy
     
  end function IO_Formats_set_ratio
  
  
  function IO_Formats_set_U(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror
    type(Job_info_Type_p) :: job_p

    type(object) :: val_obj
    real(kind=cp) :: val
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "set_U expects exactly 2 argument")
       !self, new_value
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_p)

    ierror = args%getitem(val_obj, 1) !get the value
    ierror = cast_nonstrict(val, val_obj)

    job_p%p%U = val
    
    ierror = dict_create(retval)
    r = retval%get_c_ptr()

    call args%destroy
    call val_obj%destroy
     
  end function IO_Formats_set_U
  
  
  function IO_Formats_set_V(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror

    type(Job_info_Type_p) :: job_p

    type(object) :: val_obj
    real(kind=cp) :: val
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "set_V expects exactly 2 argument")
       !self, new_value
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_p)

    ierror = args%getitem(val_obj, 1) !get the value
    ierror = cast_nonstrict(val, val_obj)

    job_p%p%V = val
    
    ierror = dict_create(retval)
    r = retval%get_c_ptr()

    call args%destroy
    call val_obj%destroy
     
  end function IO_Formats_set_V
  
  
  function IO_Formats_set_W(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror

    type(Job_info_Type_p) :: job_p

    type(object) :: val_obj
    real(kind=cp) :: val
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "set_W expects exactly 2 argument")
       !self, new_value
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_p)

    ierror = args%getitem(val_obj, 1) !get the value
    ierror = cast_nonstrict(val, val_obj)

    job_p%p%W = val
    
    ierror = dict_create(retval)
    r = retval%get_c_ptr()

    call args%destroy
    call val_obj%destroy
   
     
  end function IO_Formats_set_W
  
  
  function IO_Formats_set_X(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror

    type(Job_info_Type_p) :: job_p

    type(object) :: val_obj
    real(kind=cp) :: val
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "set_X expects exactly 2 argument")
       !self, new_value
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_p)

    ierror = args%getitem(val_obj, 1) !get the value
    ierror = cast_nonstrict(val, val_obj)

    job_p%p%X = val
    
    ierror = dict_create(retval)
    r = retval%get_c_ptr()

    call args%destroy
    call val_obj%destroy
     
  end function IO_Formats_set_X
  
  
  function IO_Formats_set_Y(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror

    type(Job_info_Type_p) :: job_p

    type(object) :: val_obj
    real(kind=cp) :: val
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "set_Y expects exactly 2 argument")
       !self, new_value
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_p)

    ierror = args%getitem(val_obj, 1) !get the value
    ierror = cast_nonstrict(val, val_obj)

    job_p%p%Y = val
    
    ierror = dict_create(retval)
    r = retval%get_c_ptr()

    call args%destroy
    call val_obj%destroy
     
  end function IO_Formats_set_Y
  
  
  function IO_Formats_set_theta_step(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror

    type(Job_info_Type_p) :: job_p

    type(object) :: val_obj
    real(kind=cp) :: val
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "set_theta_step expects exactly 2 argument")
       !self, new_value
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_p)

    ierror = args%getitem(val_obj, 1) !get the value
    ierror = cast_nonstrict(val, val_obj)

    job_p%p%theta_step = val
    
    ierror = dict_create(retval)
    r = retval%get_c_ptr()

    call args%destroy
    call val_obj%destroy
    
     
  end function IO_Formats_set_theta_step
  
  
  function IO_Formats_set_bkg(self_ptr, args_ptr) result(r) bind(c)
        
    type(c_ptr), value :: self_ptr
    type(c_ptr), value :: args_ptr
    type(c_ptr) :: r
    type(tuple) :: args
    type(dict) :: retval
    integer :: num_args
    integer :: ierror

    type(Job_info_Type_p) :: job_p

    type(object) :: val_obj
    real(kind=cp) :: val
  
    r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
    ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
    call unsafe_cast_from_c_ptr(args, args_ptr)
    ! Check if the arguments are OK
    ierror = args%len(num_args)
    ! we should also check ierror, but this example does not do complete error checking for simplicity
    if (num_args /= 2) then
       call raise_exception(TypeError, "set_bkg expects exactly 2 argument")
       !self, new_value
       call args%destroy
       return
    endif
    
    ! Doing boring stuff
    call get_job_info_type_from_arg(args, job_p)

    ierror = args%getitem(val_obj, 1) !get the value
    ierror = cast_nonstrict(val, val_obj)

    job_p%p%bkg = val
    
    ierror = dict_create(retval)
    r = retval%get_c_ptr()

    call args%destroy
    call val_obj%destroy
     
  end function IO_Formats_set_bkg

end module API_IO_Formats
