! **************************************************************************
!
! CrysFML API
!
! @file      Src/Extensions/powder_generation.f90
! @brief     Powder pattern simulation Fortran binding
!
! @homepage  https://code.ill.fr/scientific-software/crysfml
! @license   GNU LGPL (see LICENSE)
! @copyright Institut Laue Langevin 2020-now
! @authors   Scientific Computing Group at ILL (see AUTHORS), based on Elias Rabel work for Forpy, and Program_Examples/PowderPattern files
!
! **************************************************************************

module powder_generation
use forpy_mod
use, intrinsic :: iso_c_binding
use, intrinsic :: iso_fortran_env

use CFML_API_PowderSimulation

implicit none

PRIVATE
type(PythonModule), save :: mod_def
type(PythonMethodTable), save :: method_table

CONTAINS

!! Initialisation function for Python 3
function PyInit_powder_generation() bind(c, name="PyInit_powder_generation") result(m)
  !DEC$ ATTRIBUTES DLLEXPORT :: PyInit_powder_generation
  type(c_ptr) :: m
  m = init()
end function

! Initialisation function for Python 2
!subroutine init_powder_generation() bind(c, name="initpowder_generation")
!  type(c_ptr) :: m
!  m = init()
!end subroutine

! Initialisation function
function init() result(m)
  type(c_ptr) :: m
  integer :: ierror
  ierror = forpy_initialize()
  call method_table%init(1)
  call method_table%add_method("crysfml_powder_compute_powder_pattern", &                  ! method name
                               "Computes powder pattern for a given cif or cfl file", &  !doc-string
                               METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
                               c_funloc(crysfml_powder_compute_powder_pattern))  ! address of Fortran function to add
  m = mod_def%init("powder_generation", "A Python extension for crysFML powder pattern", method_table)
end function



! Implementation of our Python methods
function crysfml_powder_compute_powder_pattern(self_ptr, args_ptr) result(r) bind(c)

  ! Forpy variables
  type(c_ptr), value :: self_ptr
  type(c_ptr), value :: args_ptr
  type(c_ptr) :: r
  type(tuple) :: args
  type(dict) :: retval
  integer :: num_args
  integer :: ierror
  ! Output arguments
  type(ndarray) :: powder_pattern_array
  ! Input arguments
  type(object) :: input_filename_object, mode_object, title_object
  character(len=:), allocatable :: input_filename, mode, title
  Type(Diffraction_Pattern_Type) :: Pat
  Type(PowPat_CW_Conditions) :: PPC

  r = C_NULL_PTR   ! in case of an exception return C_NULL_PTR
  ! use unsafe_cast_from_c_ptr to cast from c_ptr to tuple
   call unsafe_cast_from_c_ptr(args, args_ptr)
 ! Check if the arguments are OK
  ierror = args%len(num_args)    ! we should also check ierror, but this example does not do complete error checking for simplicity
  if (num_args /= 14) then
    call raise_exception(TypeError, "crysfml_powder_compute_powder_pattern function expects exactly 14 arguments")
    call args%destroy
    return
  endif

  ! File to process
  ierror = args%getitem(input_filename_object, 0)
  ierror = cast_nonstrict(input_filename, input_filename_object)
  ! Mode ("CFL" "CF2" or "CIF")
  ierror = args%getitem(mode_object, 1)
  ierror = cast_nonstrict(mode, mode_object)
  ! "Title"
  ierror = args%getitem(title_object, 2)
  ierror = cast_nonstrict(title, title_object)
  PPC%title=trim(title)
  ! Lambda
  ierror = args%getitem(PPC%Lambda, 3)
  ! Resolution in u
  ierror = args%getitem(PPC%U, 4)
  ! Resolution in v
  ierror = args%getitem(PPC%V, 5)
  ! Resolution in w
  ierror = args%getitem(PPC%W, 6)
  ! Resolution in x
  ierror = args%getitem(PPC%X, 7)
  ! Theta min
  ierror = args%getitem(PPC%Thmin, 8)
  ! Theta step
  ierror = args%getitem(PPC%step, 9)
  ! Theta max
  ierror = args%getitem(PPC%Thmax, 10)
  ! Job type (0: Xrays, 1:Neutrons)
  ierror = args%getitem(PPC%job, 11)
  ! Lorentzian size
  ierror = args%getitem(PPC%Ls, 12)
  ! Background level
  ierror = args%getitem(PPC%bkg, 13)

  !
  call compute_powder_pattern(input_filename, mode, PPC, PAT)

  !
  ierror = ndarray_create(powder_pattern_array, PAT%ycalc)
  ierror = dict_create(retval)
  ierror = retval%setitem("Powder pattern", powder_pattern_array)
  ierror = retval%setitem("xmin", PAT%xmin)
  ierror = retval%setitem("xmax", PAT%xmax)
  ierror = retval%setitem("xstep", PAT%step)
  r = retval%get_c_ptr()

end function

end module
