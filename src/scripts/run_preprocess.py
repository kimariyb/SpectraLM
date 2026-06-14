import pandas as pd
import uuid
import pickle
from tqdm import tqdm

from utils.spec_utils import canonicalize_smiles, smiles_to_selfies, safe_literal_eval, process_13c_peaks, process_1h_peaks


def row_to_spectra(row):
    smiles = row["SMILES"]
    canonical_smiles = canonicalize_smiles(smiles)

    if canonical_smiles is None:
        return None

    selfies = smiles_to_selfies(canonical_smiles)
    if selfies is None:
        return None

    c_raw = safe_literal_eval(row["NMR_processed_13C"])
    h_raw = safe_literal_eval(row["NMR_processed_1H"])


    c_peaks = process_13c_peaks(c_raw)
    h_peaks = process_1h_peaks(h_raw)


    sample = {

        "id": str(uuid.uuid4()),

        # Molecule
        "smiles": smiles,
        "canonical_smiles": canonical_smiles,
        "selfies": selfies,

        # Metadata
        "meta": {
            "source": "experimental"
        },

        # 13C NMR
        "13C_NMR": {
            "frequency": row.get("NMR_frequency_13C"),
            "solvent": row.get("NMR_solvent_13C"),
            "peaks": c_peaks
        },

        # 1H NMR
        "1H_NMR": {
            "frequency": row.get("NMR_frequency_1H"),
            "solvent": row.get("NMR_solvent_1H"),
            "peaks": h_peaks
        },

        # Future spectrum image path
        "spectrum": {
            "1H_image": None,
            "13C_image": None,
            "combined_image": None
        }
    }

    return sample


def build_spectra_dataset(df):
    dataset = []

    for _, row in tqdm(df.iterrows(), desc="Processing rows"):

        try:
            sample = row_to_spectra(row)
            if sample is not None:
                dataset.append(sample)

        except Exception as e:
            print(f"  错误: {e}")
            break  # 找到第一个问题行后停止

    return dataset


if __name__ == "__main__":
    # 读取数据
    df = pd.read_csv("./dataset/NMRexp_10to24_1_1004.csv")

    # 提取两种NMR
    df_13C = df[df['NMR_type'] == '13C NMR'].copy()
    df_1H = df[df['NMR_type'] == '1H NMR'].copy()

    # 给列名加后缀，避免合并后混淆
    df_13C = df_13C.add_suffix('_13C')
    df_1H = df_1H.add_suffix('_1H')

    # 按 SMILES 合并（inner join = 只保留同时有两种 NMR 的分子）
    df_common = pd.merge(
        df_13C, df_1H,
        left_on='SMILES_13C',
        right_on='SMILES_1H',
        how='inner'
    )

    # 新增优先级列：13C 和 1H 溶剂一致的行，优先级=1，否则=0
    df_common['solvent_match_priority'] = (
        df_common['NMR_solvent_13C'] == df_common['NMR_solvent_1H']
    ).astype(int)

    # 排序：先按 SMILES 分组，再按溶剂优先级降序，确保匹配的行排在最前
    df_common = df_common.sort_values(
        by=['SMILES_13C', 'solvent_match_priority'],
        ascending=[True, False]
    )

    # 去重：每个 SMILES 只保留第一行
    df_common = df_common.drop_duplicates(subset=['SMILES_13C'], keep='first')

    # 清理临时列，统一 SMILES 列名
    df_common['SMILES'] = df_common['SMILES_13C']
    df_common.drop(
        ['SMILES_13C', 'SMILES_1H', 'solvent_match_priority'],
        axis=1,
        inplace=True
    )

    spectra_list = build_spectra_dataset(df_common)

    print("=" * 80)
    print(spectra_list[0]["1H_NMR"])
    print("=" * 80)
    print(spectra_list[0]["13C_NMR"])

    # 保存数据
    with open("./dataset/NMRexp_spectra_dataset.pkl", "wb") as f:
        pickle.dump(spectra_list, f)
    
    print(f"已保存数据！数据集大小: {len(spectra_list)}")

"""
{
    'id': '1253eab9-1f7c-4f75-8818-58af4ff14316', 
    'smiles': 'B(/C(/C)=C/C)(O)O', 
    'canonical_smiles': 'C/C=C(\\C)B(O)O', 
    'selfies': '[C][/C][=C][Branch1][C][\\C][B][Branch1][C][O][O]', 
    'meta': {
        'source': 'experimental'
    }, 
    '13C_NMR': {
        'frequency': '101 MHz', 
        'solvent': 'CDCl3', 
        'peaks': [
            {'shift': 143.8}, {'shift': 14.9}, {'shift': 12.8}
        ]
    }, 
    '1H_NMR': {
        'frequency': '400 MHz', 
        'solvent': 'CDCl3', 
        'peaks': [{'shift': 6.76, 'multiplicity': 'qd', 'J': [], 'integration': 1.0}, {'shift': 1.72, 'multiplicity': 'dd', 'J': [], 'integration': 3.0}, {'shift': 1.68, 'multiplicity': 's', 'J': [], 'integration': 3.0}]
    }, 
    'spectrum': {
        '1H_image': None, 
        '13C_image': None, 
        'combined_image': None
    }
}
"""
