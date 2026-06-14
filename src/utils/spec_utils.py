import numpy as np
import re
import ast
import selfies as sf
import matplotlib.ticker as ticker

from rdkit import Chem


def lorentzian(x: np.ndarray, x0: float, lw: float) -> np.ndarray:
    """
    Lorentzian 线型（半高宽 = lw）
    """
    return (lw / 2) ** 2 / ((x - x0) ** 2 + (lw / 2) ** 2)


def gaussian(x: np.ndarray, x0: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    
 
def pseudo_voigt(x: np.ndarray, x0: float, lw: float, eta: float = 0.5) -> np.ndarray:
    """
    Pseudo-Voigt：Lorentzian 与 Gaussian 的线性混合
    """
    sigma = lw / (2 * np.sqrt(2 * np.log(2)))
    return eta * lorentzian(x, x0, lw) + (1 - eta) * gaussian(x, x0, sigma)


def add_noise(y: np.ndarray, snr: float = 80.0, rng=None) -> np.ndarray:
    """
    叠加高斯白噪声，snr 为信噪比
    """
    if rng is None:
        rng = np.random.default_rng(42)
    peak = np.max(np.abs(y))
    sigma_noise = peak / snr
    return y + rng.normal(0, sigma_noise, size=y.shape)


def set_spectra_axes(ax, ppm_min: float, ppm_max: float):
    """统一坐标轴样式"""
    ax.set_xlim(ppm_max, ppm_min)          # NMR 惯例：从右到左
    ax.set_xlabel("Chemical Shift (ppm)", fontsize=16, labelpad=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', direction='out', length=4, labelsize=14)
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20 if ppm_max > 100 else 1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5 if ppm_max > 100 else 0.2))


def parse_integration(integ_str: str) -> float:
    """将 '3H', '2H' 等字符串转为数字"""
    m = re.search(r'[\d.]+', str(integ_str))
    return float(m.group()) if m else 1.0
 

def safe_literal_eval(x) -> list:
    """
    安全解析字符串格式的 list/dict
    """
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x


def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        return Chem.MolToSmiles(
            mol,
            canonical=True
        )

    except Exception:
        return None


def smiles_to_selfies(smiles):
    """
    smiles -> selfies
    """
    try:
        return sf.encoder(smiles)
    except Exception:
        return None


def process_13c_peaks(c_raw) -> list[dict]:
    """
    处理 13C NMR 数据

    输入格式:
    [
        (167.2, None, None),
        (33.8, 32.8, None)
    ]

    输出格式:
    [
        {
            "shift": 167.2
        },
        {
            "shift": [33.8, 32.8]
        }
    ]
    """
    peaks = []

    for item in c_raw:
        if len(item) == 0:
            continue

        # 收集所有非 None 的位移值
        shifts = []
        for val in item:
            if val is None:
                continue
            try:
                shifts.append(float(val))
            except (TypeError, ValueError):
                continue

        if not shifts:
            continue

        # 单峰存 float，重叠峰存 list
        peaks.append({
            "shift": shifts[0] if len(shifts) == 1 else shifts
        })

    return peaks


def parse_couplings(couplings: list) -> list[float]:
    """
    解析 J coupling
    """
    if couplings is None:
        return []

    if not isinstance(couplings, list):
        return []

    parsed = []
    for j in couplings:
        if isinstance(j, str):
            cleaned = re.search(r'[\d.]+', j)
            if cleaned:
                parsed.append(float(cleaned.group()))
        elif isinstance(j, (int, float)):
            parsed.append(float(j))

    return parsed


def process_1h_peaks(h_raw) -> list[dict]:
    """
    处理 1H NMR 数据

    输入格式:
    ('dd', ['5.0Hz', '3.0Hz'], '1H', 7.31, 7.31)
    ('d', ['7.8Hz'], '1H', 7.31, 7.31)
    ('s', [], '2H', 9.36, 9.36)

    meaning:
        0: multiplicity
        1: couplings
        2: num_h
        3: start_shift
        4: end_shift

    输出格式:
    [
        {
            "shift": 7.31,
            "multiplicity": "dd",
            "J": [5.0, 3.0],
            "integration": 1.0
        },
        {
            "shift": 7.31,
            "multiplicity": "d",
            "J": [7.8],
            "integration": 1.0
        },
        {
            "shift": 9.36,
            "multiplicity": "s",
            "J": [],
            "integration": 1.0
        }
    ]
    """

    peaks = []

    for item in h_raw:

        if len(item) < 5:
            continue
        
        multiplicity = item[0]
        couplings = parse_couplings(item[1])
        integration = parse_integration(item[2])
        start_shift = float(item[3])
        end_shift = float(item[4])

        center_shift = (start_shift + end_shift) / 2

        peaks.append({
            "shift": center_shift,
            "multiplicity": multiplicity,
            "J": couplings,
            "integration": integration
        })

    return peaks
