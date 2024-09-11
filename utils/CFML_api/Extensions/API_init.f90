! **************************************************************************
!
! CrysFML API
!
! @file      Src/Extensions/API_init.f90
! @brief     Initialisation functions for Fortran binding
!
! @homepage  https://code.ill.fr/scientific-software/crysfml
! @license   GNU LGPL (see LICENSE)
! @copyright Institut Laue Langevin 2020-now
! @authors   Scientific Computing Group at ILL (see AUTHORS), based on Elias Rabel work for Forpy
!
! **************************************************************************

module API_init
  use forpy_mod
  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env
  
  use API_Error_Messages, only: error_messages

  use API_Crystallographic_Symmetry,only: &
       crystallographic_symmetry_set_spacegroup, &
       crystallographic_symmetry_del_spacegroup, &
       crystallographic_symmetry_write_spacegroup, &
       crystallographic_symmetry_get_latt_trans, &
       crystallographic_symmetry_get_hexa, &
       crystallographic_symmetry_get_numspg, &
       crystallographic_symmetry_get_spg_symb, &
       crystallographic_symmetry_get_hall, &
       crystallographic_symmetry_get_ghall, &
       crystallographic_symmetry_get_CrystalSys, &
       crystallographic_symmetry_get_Laue, &
       crystallographic_symmetry_get_PG, &
       crystallographic_symmetry_get_Info, &
       crystallographic_symmetry_get_SG_setting, &
       crystallographic_symmetry_get_SPG_lat, &
       crystallographic_symmetry_get_SPG_latsy, &
       crystallographic_symmetry_get_NumLat, &
       crystallographic_symmetry_get_bravais, &
       crystallographic_symmetry_get_centre, &
       crystallographic_symmetry_get_centred, &
       crystallographic_symmetry_get_centre_coord, &
       crystallographic_symmetry_get_numops, &
       crystallographic_symmetry_get_multip, &
       crystallographic_symmetry_get_num_gen, &
       crystallographic_symmetry_get_SymOp, &
       crystallographic_symmetry_get_SymopSymb, &
       crystallographic_symmetry_get_wyckoff, &
       crystallographic_symmetry_get_R_asym_unit, &
       crystallographic_symmetry_get_symmetry_operator_rotation_matrix, &
       crystallographic_symmetry_get_symmetry_operator_trans_matrix, &
       crystallographic_symmetry_get_wyckoff_num_orbit, &
       crystallographic_symmetry_get_wyckoff_orbits, &
       crystallographic_symmetry_get_wyckoff_multp, &
       crystallographic_symmetry_get_wyckoff_site, &
       crystallographic_symmetry_get_wyckoff_norb, &
       crystallographic_symmetry_get_wyckoff_str_orig, &
       crystallographic_symmetry_get_wyckoff_str_orbit, &
       crystallographic_symmetry_get_multip_pos_crys, &
       crystallographic_symmetry_get_occ_site
       
  use API_IO_Formats, only: &
       IO_formats_readn_set_xtal_structure, &
       IO_Formats_jobinfo_from_CIF_string_array, &
       IO_Formats_del_jobinfo, &
       IO_Formats_get_title, &
       IO_Formats_get_num_phases, &
       IO_Formats_get_num_patterns, &
       IO_Formats_get_num_cmd, &
       IO_Formats_get_patt_typ, &
       IO_Formats_set_patt_typ, &
       IO_Formats_get_phas_nam, &
       IO_Formats_get_cmd, &
       IO_Formats_get_range_stl, &
       IO_Formats_get_range_q, &
       IO_Formats_get_range_d, &
       IO_Formats_get_range_2theta, &
       IO_Formats_set_range_2theta, &
       IO_Formats_get_range_energy, &
       IO_Formats_get_lambda, &
       IO_formats_set_lambda, &
       IO_Formats_get_ratio, &
       IO_Formats_set_ratio, &
       IO_Formats_get_dtt1, &
       IO_Formats_get_dtt2, &
       IO_Formats_get_U, &
       IO_Formats_get_V, &
       IO_Formats_get_W, &
       IO_Formats_get_X, &
       IO_Formats_get_Y, &
       IO_Formats_get_theta_step, &
       IO_Formats_get_bkg, &
       IO_Formats_set_U, &
       IO_Formats_set_V, &
       IO_Formats_set_W, &
       IO_Formats_set_X, &
       IO_Formats_set_Y, &
       IO_Formats_set_theta_step, &
       IO_Formats_set_bkg

  use API_Structure_Factors, only: &
       structure_factors_structure_factors, &
       structure_factors_write_structure_factors, &
       create_table_af0_xray_fun

  use API_Crystal_Metrics, only: &
       crystal_metrics_set_crystal_cell, &
       crystal_metrics_del_crystal_cell, &
       crystal_metrics_write_crystal_cell, &
       crystal_metrics_get_cell, &
       crystal_metrics_get_ang, &
       crystal_metrics_get_lcell, &
       crystal_metrics_get_lang, &
       crystal_metrics_get_cell_std, &
       crystal_metrics_get_ang_std, &
       crystal_metrics_get_rcell, &
       crystal_metrics_get_rang, &
       crystal_metrics_get_GD, &
       crystal_metrics_get_GR, &
       crystal_metrics_get_Cr_Orth_cel, &
       crystal_metrics_get_Orth_Cr_Cel, &
       crystal_metrics_get_BL_M, &
       crystal_metrics_get_BL_Minv, &
       crystal_metrics_get_cellvol, &
       crystal_metrics_get_rcellvol, &
       crystal_metrics_get_stdvol, &
       crystal_metrics_get_CartType, &
       crystal_metrics_cart_vector, &
       crystal_metrics_cart_u_vector, &
       crystal_metrics_get_betas_from_biso

  use API_Atom_TypeDef, only: &
       atom_typedef_del_atom_list, &
       atom_typedef_atomlist_from_CIF_string_array, &
       atom_typedef_atomlist_reset_occ_cif, &
       atom_typedef_atomlist_set_all_adp_cif, &
       atom_typedef_get_item, &
       atom_typedef_get_natoms, &
       atom_typedef_atom_from_string, &
       atom_typedef_del_atom, &
       atom_typedef_get_Lab, &
       atom_typedef_get_ChemSymb, &
       atom_typedef_get_SfacSymb, &
       atom_typedef_get_wyck, &
       atom_typedef_get_Active, &
       atom_typedef_get_Z, &
       atom_typedef_get_Mult, &
       atom_typedef_get_X, &
       atom_typedef_get_X_Std, &
       atom_typedef_get_MX, &
       atom_typedef_get_LX, &
       atom_typedef_get_Occ, &
       atom_typedef_get_Occ_Std, &
       atom_typedef_get_MOcc, &
       atom_typedef_get_LOcc, &
       atom_typedef_get_Biso, &
       atom_typedef_get_Biso_std, &
       atom_typedef_get_MBiso, &
       atom_typedef_get_LBiso, &
       atom_typedef_get_Utype, &
       atom_typedef_get_ThType, &
       atom_typedef_get_U, &
       atom_typedef_get_U_std, &
       atom_typedef_get_MU, &
       atom_typedef_get_LU, &
       atom_typedef_get_Ueq, &
       atom_typedef_get_Charge, &
       atom_typedef_get_Moment, &
       atom_typedef_get_Ind, &
       atom_typedef_get_NVar, &
       atom_typedef_get_VarF, &
       atom_typedef_get_MVarF, &
       atom_typedef_get_LVarF, &
       atom_typedef_get_AtmInfo, &
       atom_typedef_get_m_xyz, &
       atom_typedef_get_sm_xyz, &
       atom_typedef_get_Mm_xyz, &
       atom_typedef_get_Lm_xyz, &
       atom_typedef_set_item, &
       atom_typedef_write_atom_list

  use API_Reflections_Utilities, only: &
       reflections_utilities_hkl_uni_reflist, &
       reflections_utilities_del_reflection_list, &
       reflections_utilities_get_nref, &
       reflections_utilities_write_reflist_info, &
       reflections_utilities_del_reflection, &
       reflections_utilities_get_item, &
       reflections_utilities_get_H, &
       reflections_utilities_get_Mult, &
       reflections_utilities_get_Fo, &
       reflections_utilities_get_Fc, &
       reflections_utilities_get_SFo, &
       reflections_utilities_get_S, &
       reflections_utilities_get_W, &
       reflections_utilities_get_Phase, &
       reflections_utilities_get_A, &
       reflections_utilities_get_B

  use API_Diffraction_Patterns, only: &
       diffraction_patterns_compute_powder_pattern, &
       diffraction_patterns_del_powder_pattern, &
       diffraction_patterns_get_title, &
       diffraction_patterns_get_diff_kind, &
       diffraction_patterns_get_scat_var, &
       diffraction_patterns_get_xax_text, &
       diffraction_patterns_get_yax_text, &
       diffraction_patterns_get_instr, &
       diffraction_patterns_get_filename, &
       diffraction_patterns_get_filepath, &
       diffraction_patterns_get_xmin, &
       diffraction_patterns_get_xmax, &
       diffraction_patterns_get_ymin, &
       diffraction_patterns_get_ymax, &
       diffraction_patterns_get_scal, &
       diffraction_patterns_get_monitor, &
       diffraction_patterns_get_norm_mon, &
       diffraction_patterns_get_col_time, &
       diffraction_patterns_get_step, &
       diffraction_patterns_get_zerop, &
       diffraction_patterns_get_Tsamp, &
       diffraction_patterns_get_Tset, &
       diffraction_patterns_get_npts, &
       diffraction_patterns_get_ct_step, &
       diffraction_patterns_get_gy, &
       diffraction_patterns_get_gycalc, &
       diffraction_patterns_get_gbgr, &
       diffraction_patterns_get_gsigma, &
       diffraction_patterns_get_sig_var, &
       diffraction_patterns_get_al_x, &
       diffraction_patterns_get_al_y, &
       diffraction_patterns_get_al_ycalc, &
       diffraction_patterns_get_al_bgr, &
       diffraction_patterns_get_al_sigma, &
       diffraction_patterns_get_al_istat, &
       diffraction_patterns_get_conv, &
       diffraction_patterns_get_x, &
       diffraction_patterns_get_y, &
       diffraction_patterns_get_sigma, &
       diffraction_patterns_get_istat, &
       diffraction_patterns_get_ycalc, &
       diffraction_patterns_get_bgr, &
       diffraction_patterns_get_nd
  
  implicit none

  PRIVATE
  type(PythonModule), save      :: mod_def
  type(PythonMethodTable), save :: method_table

CONTAINS
  ! Initialisation function for Python 3
  function PyInit_crysfml_api() bind(c, name="PyInit_crysfml_api") result(m)
  !DEC$ ATTRIBUTES DLLEXPORT :: PyInit_crysfml_API
    type(c_ptr) :: m
    m = init()
  end function PyInit_crysfml_api

  ! Initialisation function for Python 2
  subroutine init_crysfml_api() bind(c, name="init_crysfml_api")
    type(c_ptr) :: m
    m = init()
  end subroutine init_crysfml_api

  ! Initialisation function
  function init() result(m)
    type(c_ptr) :: m
    integer :: ierror
    ierror = forpy_initialize()



    !--------------------------
    !Total number of method in the binding
    !--------------------------
    call method_table%init(210)



    
    !--------------------------
    ! Error Messages (1)
    !--------------------------
    call method_table%add_method("error_messages", &                  ! method name
         "gets CrysFML error messages", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(error_messages))  ! address of Fortran function to add



    
    !--------------------------
    ! Diffraction Patterns (43)
    !--------------------------
    
    call method_table%add_method("diffraction_patterns_compute_powder_pattern", &                  ! method name
         "compute the powder diffraction pattern from some experimental conditions and a set of reflections", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_compute_powder_pattern))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_del_powder_pattern", &                  ! method name
         "diffraction pattern deallocation", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_del_powder_pattern))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_title", &                  ! method name
         "title getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_title))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_diff_kind", &                  ! method name
         "diff_kind getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_diff_kind))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_scat_var", &                  ! method name
         "scat_var getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_scat_var))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_xax_text", &                  ! method name
         "xax_text getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_xax_text))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_yax_text", &                  ! method name
         "yax_text getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_yax_text))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_instr", &                  ! method name
         "instr getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_instr))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_filename", &                  ! method name
         "filename getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_filename))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_filepath", &                  ! method name
         "filepath getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_filepath))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_xmin", &                  ! method name
         "xmin getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_xmin))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_xmax", &                  ! method name
         "xmax getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_xmax))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_ymin", &                  ! method name
         "ymin getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_ymin))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_ymax", &                  ! method name
         "ymax getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_ymax))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_scal", &                  ! method name
         "scal getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_scal))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_monitor", &                  ! method name
         "monitor getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_monitor))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_norm_mon", &                  ! method name
         "norm_mon getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_norm_mon))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_col_time", &                  ! method name
         "col_time getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_col_time))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_step", &                  ! method name
         "step getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_step))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_zerop", &                  ! method name
         "zerop getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_zerop))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_Tsamp", &                  ! method name
         "Tsamp getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_Tsamp))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_Tset", &                  ! method name
         "Tset getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_Tset))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_npts", &                  ! method name
         "npts getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_npts))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_ct_step", &                  ! method name
         "ct_step getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_ct_step))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_gy", &                  ! method name
         "gy getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_gy))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_gycalc", &                  ! method name
         "gycalc getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_gycalc))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_gbgr", &                  ! method name
         "gbgr getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_gbgr))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_gsigma", &                  ! method name
         "gsigma getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_gsigma))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_sig_var", &                  ! method name
         "sig_var getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_sig_var))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_al_x", &                  ! method name
         "al_x getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_al_x))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_al_y", &                  ! method name
         "al_y getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_al_y))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_al_ycalc", &                  ! method name
         "al_ycalc getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_al_ycalc))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_al_bgr", &                  ! method name
         "al_bgr getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_al_bgr))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_al_sigma", &                  ! method name
         "al_sigma getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_al_sigma))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_al_istat", &                  ! method name
         "al_istat getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_al_istat))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_conv", &                  ! method name
         "conv getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_conv))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_x", &                  ! method name
         "x getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_x))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_y", &                  ! method name
         "y getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_y))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_sigma", &                  ! method name
         "sigma getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_sigma))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_istat", &                  ! method name
         "istat getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_istat))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_ycalc", &                  ! method name
         "ycalc getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_ycalc))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_bgr", &                  ! method name
         "bgr getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_bgr))  ! address of Fortran function to add

    call method_table%add_method("diffraction_patterns_get_nd", &                  ! method name
         "nd getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(diffraction_patterns_get_nd))  ! address of Fortran function to add


    
    !--------------------------
    ! Crystallographic Symmetry (39)
    !--------------------------
    
    call method_table%add_method("crystallographic_symmetry_set_spacegroup", &                  ! method name
         "Creates the space group", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_set_spacegroup))  ! address of Fortran function to add

    call method_table%add_method("crystallographic_symmetry_del_spacegroup", &                  ! method name
         "Delete the space group", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_del_spacegroup))  ! address of Fortran function to add

    call method_table%add_method("crystallographic_symmetry_write_spacegroup", &                  ! method name
         "Return space group description", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_write_spacegroup))  ! address of Fortran function to add

    call method_table%add_method("crystallographic_symmetry_get_latt_trans", &                  ! method name
         "latt_trans getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_latt_trans))  ! address of Fortran function to add
    
    call method_table%add_method("crystallographic_symmetry_get_hexa", &                  ! method name
         "hexa getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_hexa))  ! address of Fortran function to add

    call method_table%add_method("crystallographic_symmetry_get_numspg", &                  ! method name
         "numspg getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_numspg))  ! address of Fortran function to add

    call method_table%add_method("crystallographic_symmetry_get_spg_symb", &                  ! method name
         "spg_symb getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_spg_symb))  ! address of Fortran function to add

    call method_table%add_method("crystallographic_symmetry_get_hall", &                  ! method name
         "hall getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_hall))  ! address of Fortran function to add
    
    call method_table%add_method("crystallographic_symmetry_get_ghall", &                  ! method name
         "ghall getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_ghall))  ! address of Fortran function to add
    
    call method_table%add_method("crystallographic_symmetry_get_CrystalSys", &                  ! method name
         "CrystalSys getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_CrystalSys))  ! address of Fortran function to add
    
    call method_table%add_method("crystallographic_symmetry_get_Laue", &                  ! method name
         "Laue getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_Laue))  ! address of Fortran function to add
    
    call method_table%add_method("crystallographic_symmetry_get_PG", &                  ! method name
         "PG getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_PG))  ! address of Fortran function to add
    
    call method_table%add_method("crystallographic_symmetry_get_Info", &                  ! method name
         "Info getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_Info))  ! address of Fortran function to add
    
    call method_table%add_method("crystallographic_symmetry_get_SG_setting", &                  ! method name
         "SG_setting getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_SG_setting))  ! address of Fortran function to add
    
    call method_table%add_method("crystallographic_symmetry_get_SPG_lat", &                  ! method name
         "SPG_lat getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_SPG_lat))  ! address of Fortran function to add
    
    call method_table%add_method("crystallographic_symmetry_get_SPG_latsy", &                  ! method name
         "SPG_latsy getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_SPG_latsy))  ! address of Fortran function to add
    
    call method_table%add_method("crystallographic_symmetry_get_NumLat", &                  ! method name
         "NumLat getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_NumLat))  ! address of Fortran function to add
    
    call method_table%add_method("crystallographic_symmetry_get_bravais", &                  ! method name
         "bravais getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_bravais))  ! address of Fortran function to add
    
    call method_table%add_method("crystallographic_symmetry_get_centre", &                  ! method name
         "centre getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_centre))  ! address of Fortran function to add
    
    call method_table%add_method("crystallographic_symmetry_get_centred", &                  ! method name
         "centred getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_centred))  ! address of Fortran function to add
    
    call method_table%add_method("crystallographic_symmetry_get_centre_coord", &                  ! method name
         "centre_coord getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_centre_coord))  ! address of Fortran function to add
    
    call method_table%add_method("crystallographic_symmetry_get_numops", &                  ! method name
         "numops getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_numops))  ! address of Fortran function to add
    
    call method_table%add_method("crystallographic_symmetry_get_multip", &                  ! method name
         "multip getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_multip))  ! address of Fortran function to add
    
    call method_table%add_method("crystallographic_symmetry_get_num_gen", &                  ! method name
         "num_gen getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_num_gen))  ! address of Fortran function to add
    
    call method_table%add_method("crystallographic_symmetry_get_SymOp", &                  ! method name
         "SymOp getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_SymOp))  ! address of Fortran function to add
    
    call method_table%add_method("crystallographic_symmetry_get_SymopSymb", &                  ! method name
         "SymopSymb getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_SymopSymb))  ! address of Fortran function to add
    
    call method_table%add_method("crystallographic_symmetry_get_wyckoff", &                  ! method name
         "wyckoff getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_wyckoff))  ! address of Fortran function to add
    
    call method_table%add_method("crystallographic_symmetry_get_R_asym_unit", &                  ! method name
         "R_asym_unit getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_R_asym_unit))  ! address of Fortran function to add
    
    call method_table%add_method("crystallographic_symmetry_get_symmetry_operator_rotation_matrix", &                  ! method name
         "Symmetry operator rotation matrix getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_symmetry_operator_rotation_matrix))  ! address of Fortran function to add

    call method_table%add_method("crystallographic_symmetry_get_symmetry_operator_trans_matrix", &                  ! method name
         "Symmetry operator translation matrix getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_symmetry_operator_trans_matrix))  ! address of Fortran function to add

    call method_table%add_method("crystallographic_symmetry_get_wyckoff_num_orbit", &                  ! method name
         "Wyckoff Type num orbit getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_wyckoff_num_orbit))  ! address of Fortran function to add

    call method_table%add_method("crystallographic_symmetry_get_wyckoff_orbits", &                  ! method name
         "Wyckoff Type orbit getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_wyckoff_orbits))  ! address of Fortran function to add

    call method_table%add_method("crystallographic_symmetry_get_wyckoff_multp", &                  ! method name
         "Wyckoff Orbit multiplicity getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_wyckoff_multp))  ! address of Fortran function to add

    call method_table%add_method("crystallographic_symmetry_get_wyckoff_site", &                  ! method name
         "Wyckoff Orbit site getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_wyckoff_site))  ! address of Fortran function to add

    call method_table%add_method("crystallographic_symmetry_get_wyckoff_norb", &                  ! method name
         "Wyckoff Orbit norb getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_wyckoff_norb))  ! address of Fortran function to add

    call method_table%add_method("crystallographic_symmetry_get_wyckoff_str_orig", &                  ! method name
         "Wyckoff Orbit str_orig getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_wyckoff_str_orig))  ! address of Fortran function to add

    call method_table%add_method("crystallographic_symmetry_get_wyckoff_str_orbit", &                  ! method name
         "Wyckoff Orbit str_orbit getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_wyckoff_str_orbit))  ! address of Fortran function to add

    call method_table%add_method("crystallographic_symmetry_get_multip_pos_crys", &                  ! method name
         "Mutiplicity of an x,y,z point for a given Space Group", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_multip_pos_crys))  ! address of Fortran function to add
    call method_table%add_method("crystallographic_symmetry_get_occ_site", &                  ! method name
         "Occupancy factor of an x,y,z point for a given Space Group", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystallographic_symmetry_get_occ_site))  ! address of Fortran function to add


    !--------------------------
    ! IO formats (37)
    !--------------------------
    call method_table%add_method("IO_Formats_readn_set_xtal_structure", &                  ! method name
         "read an input file and construct the crystal structure in terms of Cell, SpG and A", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_readn_set_xtal_structure))  ! address of Fortran function to add

    call method_table%add_method("IO_Formats_del_jobinfo", &                  ! method name
         "Delete a Job info", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_del_jobinfo))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_jobinfo_from_CIF_string_array", & ! method name
         "Create a job info from an array of CIF lines", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_jobinfo_from_CIF_string_array))  ! address of Fortran function to add
       
    call method_table%add_method("IO_Formats_get_title", &                  ! method name
         "title getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_get_title))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_get_num_phases", &                  ! method name
         "num_phases getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_get_num_phases))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_get_num_patterns", &                  ! method name
         "num_patterns getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_get_num_patterns))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_get_num_cmd", &                  ! method name
         "num_cmd getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_get_num_cmd))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_get_patt_typ", &                  ! method name
         "pat_typ getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_get_patt_typ))  ! address of Fortran function to add

    call method_table%add_method("IO_Formats_set_patt_typ", &                  ! method name
         "pat_typ setter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_set_patt_typ))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_get_phas_nam", &                  ! method name
         "phas_nam getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_get_phas_nam))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_get_cmd", &                  ! method name
         "cmd getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_get_cmd))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_get_range_stl", &                  ! method name
         "range_stl getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_get_range_stl))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_get_range_q", &                  ! method name
         "range_q getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_get_range_q))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_get_range_d", &                  ! method name
         "range_d getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_get_range_d))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_get_range_2theta", &                  ! method name
         "range_2theta getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_get_range_2theta))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_set_range_2theta", &                  ! method name
         "range_2theta setter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_set_range_2theta))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_get_range_energy", &                  ! method name
         "range_energy getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_get_range_energy))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_get_lambda", &                  ! method name
         "lambda getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_get_lambda))  ! address of Fortran function to add

    call method_table%add_method("IO_Formats_set_lambda", &                  ! method name
         "lambda getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_set_lambda))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_get_ratio", &                  ! method name
         "ratio getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_get_ratio))  ! address of Fortran function to add

    call method_table%add_method("IO_Formats_set_ratio", &                  ! method name
         "ratio setter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_set_ratio))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_get_dtt1", &                  ! method name
         "dtt1 getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_get_dtt1))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_get_dtt2", &                  ! method name
         "dtt2 getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_get_dtt2))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_get_U", &                  ! method name
         "U getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_get_U))  ! address of Fortran function to add

    call method_table%add_method("IO_Formats_set_U", &                  ! method name
         "U setter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_set_U))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_get_V", &                  ! method name
         "V getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_get_V))  ! address of Fortran function to add

    call method_table%add_method("IO_Formats_set_V", &                  ! method name
         "V setter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_set_V))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_get_W", &                  ! method name
         "W getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_get_W))  ! address of Fortran function to add

    call method_table%add_method("IO_Formats_set_W", &                  ! method name
         "W setter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_set_W))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_get_X", &                  ! method name
         "X getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_get_X))  ! address of Fortran function to add

    call method_table%add_method("IO_Formats_set_X", &                  ! method name
         "X setter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_set_X))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_get_Y", &                  ! method name
         "Y getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_get_Y))  ! address of Fortran function to add

    call method_table%add_method("IO_Formats_set_Y", &                  ! method name
         "Y detter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_set_Y))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_get_theta_step", &                  ! method name
         "theta_step getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_get_theta_step))  ! address of Fortran function to add

    call method_table%add_method("IO_Formats_set_theta_step", &                  ! method name
         "theta_step setter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_set_theta_step))  ! address of Fortran function to add
    
    call method_table%add_method("IO_Formats_get_bkg", &                  ! method name
         "bkg getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_get_bkg))  ! address of Fortran function to add

    call method_table%add_method("IO_Formats_set_bkg", &                  ! method name
         "bkg setter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(IO_Formats_set_bkg))  ! address of Fortran function to add
    
    !--------------------------
    ! Atom Typedef (47)
    !--------------------------
    call method_table%add_method("atom_typedef_del_atom_list", &                  ! method name
         "Delete an atom list", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_del_atom_list))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_atomlist_from_CIF_string_array", & ! method name
         "Create an atom list from an array of CIF lines", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_atomlist_from_CIF_string_array))  ! address of Fortran function to add

    call method_table%add_method("atom_typedef_atomlist_reset_occ_cif", & ! method name
         "Set mult and occ according to Sfac convention", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_atomlist_reset_occ_cif))  ! address of Fortran function to add

    call method_table%add_method("atom_typedef_atomlist_set_all_adp_cif", & ! method name
         "Set all Atom fields for Us, Bs and betas", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_atomlist_set_all_adp_cif))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_set_item", &                  ! method name
         "Set an item in the atom list", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_set_item))  ! address of Fortran function to add

    call method_table%add_method("atom_typedef_get_item", &                  ! method name
         "Get a specific atom", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_item))  ! address of Fortran function to add

    call method_table%add_method("atom_typedef_get_natoms", &                  ! method name
         "natoms getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_natoms))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_write_atom_list", &                  ! method name
         "Return the atom list description", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_write_atom_list))  ! address of Fortran function to add

    call method_table%add_method("atom_typedef_atom_from_string", &                  ! method name
         "Create an atom from a string", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_atom_from_string))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_del_atom", &                  ! method name
         "Delete an atom", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_del_atom))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_Lab", &                  ! method name
         "Lab getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_Lab))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_ChemSymb", &                  ! method name
         "ChemSymb getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_ChemSymb))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_SfacSymb", &                  ! method name
         "SfacSymb getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_SfacSymb))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_wyck", &                  ! method name
         "wyck getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_wyck))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_Active", &                  ! method name
         "Active getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_Active))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_Z", &                  ! method name
         "Z getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_Z))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_Mult", &                  ! method name
         "Mult getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_Mult))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_X", &                  ! method name
         "X getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_X))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_X_Std", &                  ! method name
         "X_Std getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_X_Std))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_MX", &                  ! method name
         "MX getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_MX))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_LX", &                  ! method name
         "LX getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_LX))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_Occ", &                  ! method name
         "Occ getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_Occ))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_Occ_Std", &                  ! method name
         "Occ_Std getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_Occ_Std))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_MOcc", &                  ! method name
         "MOcc getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_MOcc))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_LOcc", &                  ! method name
         "LOcc getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_LOcc))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_Biso", &                  ! method name
         "Biso getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_Biso))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_Biso_std", &                  ! method name
         "Biso_std getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_Biso_std))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_MBiso", &                  ! method name
         "MBiso getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_MBiso))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_LBiso", &                  ! method name
         "LBiso getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_LBiso))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_Utype", &                  ! method name
         "Utype getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_Utype))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_ThType", &                  ! method name
         "ThType getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_ThType))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_U", &                  ! method name
         "U getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_U))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_U_std", &                  ! method name
         "U_std getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_U_std))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_MU", &                  ! method name
         "MU getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_MU))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_LU", &                  ! method name
         "LU getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_LU))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_Ueq", &                  ! method name
         "Ueq getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_Ueq))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_Charge", &                  ! method name
         "Charge getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_Charge))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_Moment", &                  ! method name
         "Moment getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_Moment))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_Ind", &                  ! method name
         "Ind getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_Ind))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_NVar", &                  ! method name
         "NVar getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_NVar))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_VarF", &                  ! method name
         "VarF getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_VarF))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_MVarF", &                  ! method name
         "MVarF getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_MVarF))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_LVarF", &                  ! method name
         "LVarF getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_LVarF))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_AtmInfo", &                  ! method name
         "AtmInfo getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_AtmInfo))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_m_xyz", &                  ! method name
         "m_xyz getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_m_xyz))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_sm_xyz", &                  ! method name
         "sm_xyz getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_sm_xyz))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_Mm_xyz", &                  ! method name
         "Mm_xyz getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_Mm_xyz))  ! address of Fortran function to add
    
    call method_table%add_method("atom_typedef_get_Lm_xyz", &                  ! method name
         "Lm_xyz getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(atom_typedef_get_Lm_xyz))  ! address of Fortran function to add


    
    !--------------------------
    ! Reflection Utilities (14)
    !--------------------------
    call method_table%add_method("reflections_utilities_hkl_uni_reflist", &                  ! method name
         "Return the list of reflections", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(reflections_utilities_hkl_uni_reflist))  ! address of Fortran function to add

    call method_table%add_method("reflections_utilities_del_reflection_list", &                  ! method name
         "Reflection list deallocation", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(reflections_utilities_del_reflection_list))  ! address of Fortran function to add

    call method_table%add_method("reflections_utilities_get_nref", &                  ! method name
         "nref getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(reflections_utilities_get_nref))  ! address of Fortran function to add

    call method_table%add_method("reflections_utilities_write_reflist_info", &                  ! method name
         "print reflist info", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(reflections_utilities_write_reflist_info))  ! address of Fortran function to add
    
    call method_table%add_method("reflections_utilities_del_reflection", &                  ! method name
         "Reflection deallocation", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(reflections_utilities_del_reflection))  ! address of Fortran function to add

    call method_table%add_method("reflections_utilities_get_item", &                  ! method name
         "Get a specific reflection", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(reflections_utilities_get_item))  ! address of Fortran function to add

    call method_table%add_method("reflections_utilities_get_H", &                  ! method name
         "H getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(reflections_utilities_get_H))  ! address of Fortran function to add
    
    call method_table%add_method("reflections_utilities_get_Mult", &                  ! method name
         "Mult getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(reflections_utilities_get_Mult))  ! address of Fortran function to add
    
    call method_table%add_method("reflections_utilities_get_Fo", &                  ! method name
         "Fo getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(reflections_utilities_get_Fo))  ! address of Fortran function to add
    
    call method_table%add_method("reflections_utilities_get_Fc", &                  ! method name
         "Fc getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(reflections_utilities_get_Fc))  ! address of Fortran function to add
    
    call method_table%add_method("reflections_utilities_get_SFo", &                  ! method name
         "SFo getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(reflections_utilities_get_SFo))  ! address of Fortran function to add
    
    call method_table%add_method("reflections_utilities_get_S", &                  ! method name
         "S getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(reflections_utilities_get_S))  ! address of Fortran function to add
    
    call method_table%add_method("reflections_utilities_get_W", &                  ! method name
         "W getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(reflections_utilities_get_W))  ! address of Fortran function to add
    
    call method_table%add_method("reflections_utilities_get_Phase", &                  ! method name
         "Phase getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(reflections_utilities_get_Phase))  ! address of Fortran function to add
    
    call method_table%add_method("reflections_utilities_get_A", &                  ! method name
         "A getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(reflections_utilities_get_A))  ! address of Fortran function to add
    
    call method_table%add_method("reflections_utilities_get_B", &                  ! method name
         "B getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(reflections_utilities_get_B))  ! address of Fortran function to add
    


    !--------------------------
    ! Structure Factors (3)
    !--------------------------
    call method_table%add_method("structure_factors_structure_factors", &                  ! method name
         "Computes structure factors", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(structure_factors_structure_factors))  ! address of Fortran function to add

    call method_table%add_method("structure_factors_write_structure_factors", &                  ! method name
         "Print structure factors", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(structure_factors_write_structure_factors))  ! address of Fortran function to add

    call method_table%add_method("create_table_af0_xray_fun", &  ! メソッド名
         "Creates Set_Fixed_Tables ", &  ! ドキュメント文字列
         METH_VARARGS, &                  ! 引数を取るがキーワード引数はなし
         c_funloc(create_table_af0_xray_fun))  ! Fortran関数のアドレスを登録

    
    !--------------------------
    ! Crystal Metrics (24)
    !--------------------------
    call method_table%add_method("crystal_metrics_set_crystal_cell", &                  ! method name
         "Creates the crystal cell", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_set_crystal_cell))  ! address of Fortran function to add
    
    call method_table%add_method("crystal_metrics_del_crystal_cell", &                  ! method name
         "Creates the crystal cell", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_del_crystal_cell))  ! address of Fortran function to add

    call method_table%add_method("crystal_metrics_write_crystal_cell", &                  ! method name
         "Return the crystal cell description", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_write_crystal_cell))  ! address of Fortran function to add

    call method_table%add_method("crystal_metrics_get_cell", &                  ! method name
         "cell getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_get_cell))  ! address of Fortran function to add
    
    call method_table%add_method("crystal_metrics_get_ang", &                  ! method name
         "angl getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_get_ang))  ! address of Fortran function to add

    call method_table%add_method("crystal_metrics_get_lcell", &                  ! method name
         "lcell getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_get_lcell))  ! address of Fortran function to add
    
    call method_table%add_method("crystal_metrics_get_lang", &                  ! method name
         "lang getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_get_lang))  ! address of Fortran function to add
    
    call method_table%add_method("crystal_metrics_get_cell_std", &                  ! method name
         "cell_std getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_get_cell_std))  ! address of Fortran function to add
    
    call method_table%add_method("crystal_metrics_get_ang_std", &                  ! method name
         "ang_std getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_get_ang_std))  ! address of Fortran function to add
    
    call method_table%add_method("crystal_metrics_get_rcell", &                  ! method name
         "rcell getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_get_rcell))  ! address of Fortran function to add
    
    call method_table%add_method("crystal_metrics_get_rang", &                  ! method name
         "rang getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_get_rang))  ! address of Fortran function to add
    
    call method_table%add_method("crystal_metrics_get_GD", &                  ! method name
         "GD getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_get_GD))  ! address of Fortran function to add
    
    call method_table%add_method("crystal_metrics_get_GR", &                  ! method name
         "GR getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_get_GR))  ! address of Fortran function to add
    
    call method_table%add_method("crystal_metrics_get_Cr_Orth_Cel", &                  ! method name
         "Cr_Orth_Cel getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_get_Cr_Orth_Cel))  ! address of Fortran function to add
    
    call method_table%add_method("crystal_metrics_get_Orth_Cr_Cel", &                  ! method name
         "Orth_Cr_Cel getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_get_Orth_Cr_Cel))  ! address of Fortran function to add
    
    call method_table%add_method("crystal_metrics_get_BL_M", &                  ! method name
         "BL_M getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_get_BL_M))  ! address of Fortran function to add
    
    call method_table%add_method("crystal_metrics_get_BL_Minv", &                  ! method name
         "BL_Minv getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_get_BL_Minv))  ! address of Fortran function to add
    
    call method_table%add_method("crystal_metrics_get_cellvol", &                  ! method name
         "cellvol getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_get_cellvol))  ! address of Fortran function to add
    
    call method_table%add_method("crystal_metrics_get_rcellvol", &                  ! method name
         "rcellvol getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_get_rcellvol))  ! address of Fortran function to add
    
    call method_table%add_method("crystal_metrics_get_stdvol", &                  ! method name
         "stdvol getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_get_stdvol))  ! address of Fortran function to add
    
    call method_table%add_method("crystal_metrics_get_CartType", &                  ! method name
         "CartType getter", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_get_CartType))  ! address of Fortran function to add

    call method_table%add_method("crystal_metrics_cart_vector", &                  ! method name
         "Convert a vector in crystal space to cartesian components", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_cart_vector))  ! address of Fortran function to add

    call method_table%add_method("crystal_metrics_cart_u_vector", &                  ! method name
         "Convert a vector in crystal space to unitary cartesian components", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_cart_u_vector))  ! address of Fortran function to add

    call method_table%add_method("crystal_metrics_get_betas_from_biso", &                  ! method name
         "Convert a vector in crystal space to cartesian components", &  !doc-string
         METH_VARARGS, &                  ! this method takes arguments but no keyword arguments
         c_funloc(crystal_metrics_get_betas_from_biso))  ! address of Fortran function to add



    
    
    m = mod_def%init("crysfml_symmetry", "A Python extension for crysFML symmetry", method_table)
  end function init


end module API_init
