import crysfml_api
from API_Reflections_Utilities import ReflectionList
from API_IO
# CIFデータからCellオブジェクトを作成（Cellは仮の名前、実際は別のAPIで作成します）
cif_data = [
    "data_NaCl",
    "_symmetry_space_group_name_H-M   'F m -3 m'",
    "_symmetry_Int_Tables_number       225",
    "_cell_length_a                   5.6400(2)",
    "_cell_length_b                   5.6400(2)",
    "_cell_length_c                   5.6400(2)",
    "_cell_angle_alpha                90",
    "_cell_angle_beta                 90",
    "_cell_angle_gamma                90",
    "_cell_volume                     179.323",
    "_exptl_crystal_density_diffrn    2.165",
    "_atom_site_label                 Na",
    "_atom_site_fract_x               0",
    "_atom_site_fract_y               0",
    "_atom_site_fract_z               0",
    "_atom_site_U_iso_or_equiv        0.02",
    "_atom_site_label                 Cl",
    "_atom_site_fract_x               0.5",
    "_atom_site_fract_y               0.5",
    "_atom_site_fract_z               0.5",
    "_atom_site_U_iso_or_equiv        0.03"
]

# 原子リストの初期化
atom_list = crysfml_api.atom_typedef_atomlist_from_CIF_string_array(cif_data)

# 空間群オブジェクトの初期化
space_group = crysfml_api.crystallographic_symmetry_set_spacegroup("F m -3 m")

# Fortranアドレスを取得（例: job_info のオブジェクトが必要）
job_info =   # ここで正しいジョブ情報のオブジェクトを設定する
key = 0  # 例として、最初のパターンタイプを取得する場合

# IO_Formats_get_patt_typを呼び出し
pattern_type = crysfml_api.IO_Formats_get_patt_typ(job_info.get_fortran_address(), key + 1)["patt_typ"]

# ReflectionListの初期化
reflection_list = ReflectionList(cell=atom_list, spg=space_group, lfriedel=True, job=job_info)