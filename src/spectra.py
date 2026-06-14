"""
该文件定义将 1H 和 13C NMR 离散数据转化为谱图的方法

NMR 数据格式说明：

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
import numpy as np
import matplotlib.pyplot as plt
import re
from math import comb
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg

try:
    from .utils.spec_utils import set_spectra_axes, add_noise, pseudo_voigt
except ImportError:
    from utils.spec_utils import set_spectra_axes, add_noise, pseudo_voigt


DPI = 100  # 标准高清DPI，不模糊
WIDTH_PX = 1200
HEIGHT_PX = 500   # 单独图高
COMB_HEIGHT = 1200   # 合并图总高


# ────────────────────────────────────────────────────────────────
# 裂分注册表：只收录频次 > 10,000 的裂分类型
# value = 各级裂分的等价质子数列表（按顺序消耗 J_list）
# 物理约定：n_equiv 个等价质子 → n_equiv+1 条线，二项式强度比，
#           相邻线间距 = J (ppm)
# ────────────────────────────────────────────────────────────────
MULT_SPLITS: dict[str, list[int]] = {
    's':    [],           # singlet
    'd':    [1],          # doublet
    't':    [2],          # triplet
    'q':    [3],          # quartet
    'p':    [4],          # quintet
    'hept': [6],          # septet
    'dd':   [1, 1],       # doublet of doublets
    'ddd':  [1, 1, 1],    # doublet of doublet of doublets
    'dt':   [1, 2],       # doublet of triplets
    'td':   [2, 1],       # triplet of doublets
    'dq':   [1, 3],       # doublet of quartets
    'ddt':  [1, 1, 2],    # doublet of doublet of triplets
    'tt':   [2, 2],       # triplet of triplets
    'qd':   [3, 1],       # quartet of doublets
    'dddd': [1, 1, 1, 1], # doublet of doublet of doublet of doublets
    # 频次 < 10,000 的类型全部在 multiplet_peaks() 里降级为 m
}

# 宽峰前缀：剥离后取 core pattern，同时放大线宽
BROAD_PREFIX = 'br'
BROAD_LW_MULT = 3.5
DEFAULT_J = 0.010   # ≈ 4 Hz @ 400 MHz


def _apply_splitting(
    positions: list[float],
    heights:   list[float],
    J:         float,
    n_equiv:   int,
) -> tuple[list[float], list[float]]:
    """
    对现有子峰做一次裂分。

    n_equiv 个等价质子，耦合常数 J (ppm)
    → 每条线裂分为 n_equiv+1 条，二项式相对强度，相邻线间距 = J。
    """
    n_lines = n_equiv + 1
    binom = [comb(n_equiv, i) for i in range(n_lines)]
    new_pos: list[float] = []
    new_h:   list[float] = []
    for p, h in zip(positions, heights):
        for i in range(n_lines):
            new_pos.append(p + (i - n_equiv / 2) * J)
            new_h.append(h * binom[i])
    return new_pos, new_h


def _parse_J(J_list: list, freq_mhz: float) -> list[float]:
    """Hz（或含单位字符串）→ ppm。"""
    result = []
    for j in J_list:
        m = re.search(r'[\d.]+', str(j))
        if m:
            result.append(float(m.group()) / freq_mhz)
    return result


def _get_J(J_ppm: list[float], idx: int) -> float:
    """安全取第 idx 个耦合常数；越界时用末尾值衰减，空列表用默认值。"""
    if not J_ppm:
        return DEFAULT_J
    if idx < len(J_ppm):
        return J_ppm[idx]
    # 次级耦合通常更小，按 0.6 衰减估算
    return J_ppm[-1] * (0.6 ** (idx - len(J_ppm) + 1))


def multiplet_peaks(shift_center: float, multiplicity: str, J_list: list, freq_mhz: float, rng):
    """
    计算多重峰的子峰位置与相对高度。

    Parameters
    ----------
    shift_center : 化学位移中心 (ppm)
    multiplicity : 裂分类型字符串（如 's', 'dd', 'ddt', 'brs', 'app t' …）
    J_list       : 耦合常数列表（Hz 或含单位字符串）；顺序与裂分级数对应
    freq_mhz     : 仪器频率 (MHz)
    rng          : numpy.random.Generator

    Returns
    -------
    positions : list[float]  各子峰 ppm 位置
    heights   : list[float]  各子峰相对高度（未归一化）
    lw_mult   : float        线宽倍数（宽峰 > 1.0）
    """
    J_ppm   = _parse_J(J_list, freq_mhz)
    lw_mult = 1.0

    # ── 归一化：小写、去空格 ──
    mult = multiplicity.lower().replace(' ', '').strip()

    # ── 处理宽峰前缀 'br' ──
    # brs → s + lw×3.5 | brd → d + lw×3.5 | br 单独 → 宽单峰
    if mult.startswith(BROAD_PREFIX):
        lw_mult = BROAD_LW_MULT
        core = mult[len(BROAD_PREFIX):]   # 剥去 'br'
        mult = core if core else 's'       # 纯 'br' 视为宽单峰

    # ── 'app xxx' 变体（apparent）：去掉前缀后当作精确裂分处理 ──
    if mult.startswith('app'):
        mult = mult[3:]

    # ── 多重峰 / 未知类型 → 宽多重峰模拟 ──
    if mult == 'm' or mult not in MULT_SPLITS:
        n_sub   = int(rng.integers(7, 15))
        span    = max(J_ppm[0] if J_ppm else 0.025, 0.015)
        sub_pos = rng.uniform(shift_center - span, shift_center + span, n_sub)
        sub_h   = rng.uniform(0.4, 1.0, n_sub)
        return sub_pos.tolist(), sub_h.tolist(), lw_mult * 2.0

    split_seq = MULT_SPLITS[mult]

    # ── 单峰 ──
    if not split_seq:
        return [shift_center], [1.0], lw_mult

    # ── 逐级裂分 ──
    positions: list[float] = [shift_center]
    heights:   list[float] = [1.0]
    for i, n_equiv in enumerate(split_seq):
        positions, heights = _apply_splitting(
            positions, heights, _get_J(J_ppm, i), n_equiv
        )

    return positions, heights, lw_mult


def compute_1H(
    data: dict,
    snr:  float,
    rng,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """
    计算 1H 谱线数组。

    Returns
    -------
    x            : ppm 轴
    y            : 归一化强度（含噪声）
    integral_data: [(shift_center, n_H), ...]
    """
    # 解析 NMR 数据中的 peaks
    nmr = data['1H_NMR']
    freq_mhz = float(re.search(r'[\d.]+', nmr['frequency']).group())

    # 计算 ppm 轴范围
    ppm_min, ppm_max = 0.0, 12.0
    x = np.linspace(ppm_min, ppm_max, 32768)
    y = np.zeros_like(x)

    # 计算线宽
    # 1H 线宽较窄（去耦后仍有 ~0.1 Hz）
    # 1H 线宽为 0.008 ± 0.004 Hz
    lw_base = 0.008
    eta = 0.55
    integral_data: list[tuple[float, float]] = []

    # 计算 1H 谱线
    for entry in nmr['peaks']:
        if isinstance(entry, dict):
            shift  = float(entry['shift'])
            mult   = str(entry.get('multiplicity', 's'))
            J_list = entry.get('J', [])
            n_H    = float(entry.get('integration', 1.0))
        else:
            shift, mult, J_list, n_H = entry
            shift, n_H = float(shift), float(n_H)

        positions, heights, lw_mult = multiplet_peaks(
            shift, mult, J_list, freq_mhz, rng
        )
        lw    = lw_base * lw_mult * float(rng.uniform(0.85, 1.15))
        h_sum = sum(heights) or 1.0
        for pos, h in zip(positions, heights):
            y += (n_H * h / h_sum) * pseudo_voigt(x, pos, lw, eta)
        integral_data.append((shift, n_H))

    y /= (y.max() or 1.0)
    y  = add_noise(y, snr=snr, rng=rng)
    y  = np.clip(y, -0.03, None)
    return x, y, integral_data


def compute_13C(
    data: dict,
    snr:  float,
    rng,
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算 13C 谱线数组。

    Returns
    -------
    x : ppm 轴（0–220 ppm）
    y : 归一化强度（含噪声）
    """
    # 解析 NMR 数据中的 peaks
    nmr = data['13C_NMR']
    freq_mhz = float(re.search(r'[\d.]+', nmr['frequency']).group())

    # 计算 ppm 轴范围
    ppm_min, ppm_max = 0.0, 220.0
    x = np.linspace(ppm_min, ppm_max, 32768)
    y = np.zeros_like(x)

    # 计算线宽
    lw_base = 0.06   # 13C 线宽较宽（去耦后仍有 ~1 Hz）
    eta = 0.60

    # 计算 13C 谱线
    for entry in nmr['peaks']:
        shift = float(entry['shift'] if isinstance(entry, dict) else entry)
        lw = lw_base * float(rng.uniform(0.8, 1.2))
        y += pseudo_voigt(x, shift, lw, eta)

    # 归一化并添加噪声
    y /= (y.max() or 1.0)
    y  = add_noise(y, snr=snr, rng=rng)
    y  = np.clip(y, -0.03, None)

    return x, y


def draw_1H(
    ax: plt.Axes,
    x:  np.ndarray,
    y:  np.ndarray,
    ppm_min: float = 0.0,
    ppm_max: float = 12.0,
    label:   str   = '',
) -> None:
    """将 1H 谱数据绘制到已有 Axes 上。"""
    # 谱线
    ax.plot(x, y, color='black', linewidth=1, zorder=3)

    # 统一坐标轴样式
    set_spectra_axes(ax, ppm_min, ppm_max)

    if label:
        ax.text(
            0.01, 0.95, label,
            transform=ax.transAxes,
            fontsize=18, va='top', color='black',
            fontweight='bold',
        )


def draw_13C(
    ax: plt.Axes,
    x:  np.ndarray,
    y:  np.ndarray,
    ppm_min: float = 0.0,
    ppm_max: float = 220.0,
    label:   str   = '',
) -> None:
    """将 13C 谱数据绘制到已有 Axes 上。"""
    ax.plot(x, y, color='black', linewidth=1, zorder=3)

    # 统一坐标轴样式
    set_spectra_axes(ax, ppm_min, ppm_max)

    if label:
        ax.text(
            0.01, 0.95, label,
            transform=ax.transAxes,
            fontsize=18, va='top', color='black',
            fontweight='bold',
        )


def HydrogenToSpectra(data: dict, snr: float = 500.0) -> Image:
    """生成独立 1H 谱图。"""
    rng = np.random.default_rng()
    x, y, _ = compute_1H(data, snr, rng)

    fig, ax = plt.subplots(
        figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI),
        dpi=DPI,
    )

    draw_1H(ax, x, y)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    buf = np.asarray(canvas.buffer_rgba())
    image = Image.fromarray(buf[..., :3])
    plt.close(fig)

    return image


def CarbonToSpectra(data: dict, snr: float = 500.0) -> Image:
    """生成独立 13C 谱图。"""
    rng = np.random.default_rng()
    x, y = compute_13C(data, snr, rng)

    fig, ax = plt.subplots(
        figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI),
        dpi=DPI,
    )
    draw_13C(ax, x, y)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    buf = np.asarray(canvas.buffer_rgba())
    image = Image.fromarray(buf[..., :3])
    plt.close(fig)

    return image


def CombineSpectra(
    data:  dict,
    h_snr: float = 500.0,
    c_snr: float = 500.0,
) -> Image:
    """
    将 1H 和 13C 谱图合并为单张图片。
    """
    rng = np.random.default_rng()

    x_h, y_h, _ = compute_1H(data, h_snr, rng)
    x_c, y_c = compute_13C(data, c_snr, rng)

    fig, (ax_h, ax_c) = plt.subplots(
        2, 1,
        figsize=(WIDTH_PX / DPI, COMB_HEIGHT / DPI),
        dpi=DPI,
    )

    draw_1H(
        ax_h, x_h, y_h,
        label='1H NMR',
    )
    draw_13C(
        ax_c, x_c, y_c,
        label='13C NMR',
    )

    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    buf = np.asarray(canvas.buffer_rgba())
    image = Image.fromarray(buf[..., :3])
    plt.close(fig)

    return image


if __name__ == '__main__':
    data = {
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



