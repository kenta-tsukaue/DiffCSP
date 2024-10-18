import pandas as pd
import re

# ファイルパスを指定
file_path = 'Data/train.csv'

# train.csv を読み込む
df = pd.read_csv(file_path, quotechar='"', escapechar='\\')

def extract_formula(cif_text):
    """
    cif テキストから _chemical_formula_structural の値を抽出する関数
    """
    match = re.search(r'_chemical_formula_structural\s+\'?\"?(\w+)\'?\"?', cif_text)
    if match:
        return match.group(1)
    else:
        return None

# 新しい列 'extracted_formula' を作成
df['extracted_formula'] = df['cif'].apply(extract_formula)

# 組成式のカウント
formula_counts = df['extracted_formula'].value_counts().reset_index()
formula_counts.columns = ['組成式', 'num']

# データフレームを作成
result_df = formula_counts[['組成式', 'num']]

# Excel ファイルとして保存
output_file = 'formula_counts.xlsx'
result_df.to_excel(output_file, index=False)

print(f"集計結果を {output_file} として保存しました。")