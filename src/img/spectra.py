"""
该文件定义将 1H 和 13C NMR 离散数据转化为谱图的方法

NMR 数据格式说明：
  13C data: [(shift_ppm, coupling_type_or_None, J_Hz_or_None), ...]
  1H data:  [(multiplicity, [J_couplings], integration, shift_max, shift_center), ...]
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import ast
import re

DPI = 100  # 标准高清DPI，不模糊
WIDTH_PX = 1508
HEIGHT_PX = 868


# ─────────────────────────────────────────────────────────────────
#  内部工具函数
# ─────────────────────────────────────────────────────────────────

def _lorentzian(x: np.ndarray, x0: float, lw: float) -> np.ndarray:
    """Lorentzian 线型（半高宽 = lw）"""
    return (lw / 2) ** 2 / ((x - x0) ** 2 + (lw / 2) ** 2)


def _gaussian(x: np.ndarray, x0: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def _pseudo_voigt(x: np.ndarray, x0: float, lw: float, eta: float = 0.5) -> np.ndarray:
    """Pseudo-Voigt：Lorentzian 与 Gaussian 的线性混合"""
    sigma = lw / (2 * np.sqrt(2 * np.log(2)))
    return eta * _lorentzian(x, x0, lw) + (1 - eta) * _gaussian(x, x0, sigma)


def _add_noise(y: np.ndarray, snr: float = 80.0, rng=None) -> np.ndarray:
    """叠加高斯白噪声，snr 为信噪比"""
    if rng is None:
        rng = np.random.default_rng(42)
    peak = np.max(np.abs(y))
    sigma_noise = peak / snr
    return y + rng.normal(0, sigma_noise, size=y.shape)


def _parse_data_string(data_str: str):
    """安全地将字符串解析为 Python 对象"""
    return ast.literal_eval(data_str)


def _style_axes(ax, ppm_min: float, ppm_max: float):
    """统一坐标轴样式"""
    ax.set_xlim(ppm_max, ppm_min)          # NMR 惯例：从右到左
    ax.set_xlabel("Chemical Shift (ppm)", fontsize=18, labelpad=6)
    ax.set_ylim(-0.1, 1 / 0.75)   # ← 加这一行，峰顶在 75% 处
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', direction='out', length=4, labelsize=14)
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20 if ppm_max > 100 else 1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5 if ppm_max > 100 else 0.2))


# ─────────────────────────────────────────────────────────────────
#  13C NMR
# ─────────────────────────────────────────────────────────────────

def CarbonToSpectra(data: dict, image_path: str = 'spectra_13C.png') -> plt.Figure:
    """
    将 13C NMR 离散数据转化为伪实验谱图并保存。

    Parameters
    ----------
    data : dict
        包含 'frequency', 'solvent', 'data' 键的字典。
        'data' 为字符串，表示 [(shift, coupling_type, J_Hz), ...] 列表。
        coupling_type 为 None 表示单峰；'d' 表示双峰等。
        J_Hz 为耦合常数（Hz），用于计算裂分峰距离。
    image_path : str
        输出图片路径。

    Returns
    -------
    matplotlib.figure.Figure
    """
    rng = np.random.default_rng(2024)
 
    # ── 解析数据 ──
    raw = _parse_data_string(data['data'])
    # freq_hz = float(re.search(r'[\d.]+', data['frequency']).group())  # MHz
    # freq_label = data['frequency']
 
    # ── 谱图参数 ──
    ppm_min, ppm_max = -10, 220.0
    x = np.linspace(ppm_min, ppm_max, 120000)
    y = np.zeros_like(x)
 
    # 线宽（ppm）；13C 线宽较宽
    lw_base = 0.1        # ppm，基础 Lorentzian 半高宽
    eta = 0.6              # Pseudo-Voigt 混合比
 
    for shift, coup_type, J_hz in raw:
        # 13C 全部视为单峰，不考虑耦合裂分
        lw = lw_base * rng.uniform(0.85, 1.15)
        y += _pseudo_voigt(x, shift, lw, eta)

    # 归一化 + 噪声
    y /= y.max()
    y = _add_noise(y, snr=150.0, rng=rng)
    y = np.clip(y, -0.05, None)

    # ── 绘图 ──
    fig, ax = plt.subplots(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI)
    ax.plot(x, y, color='black', linewidth=1, zorder=3)
    _style_axes(ax, ppm_min, ppm_max)


    
    plt.tight_layout()
    fig.savefig(image_path, dpi=DPI, bbox_inches='tight', pad_inches=0.0)
    print(f"[13C] 谱图已保存至 {image_path}")
    return fig


# ─────────────────────────────────────────────────────────────────
#  1H NMR
# ─────────────────────────────────────────────────────────────────

# 多重峰相对强度表（Pascal 三角形）
_MULTIPLET_PATTERNS = {
    's':  ([0],                    [1]),
    'd':  ([-1, 1],                [1, 1]),
    't':  ([-1, 0, 1],             [1, 2, 1]),
    'q':  ([-1.5,-0.5,0.5,1.5],   [1, 3, 3, 1]),
    'p':  ([-2,-1,0,1,2],          [1, 4, 6, 4, 1]),
    'h':  ([-2.5,-1.5,-0.5,0.5,1.5,2.5], [1,5,10,10,5,1]),
    'hept':([-3,-2,-1,0,1,2,3],   [1,6,15,20,15,6,1]),
    'm':  None,   # 多重峰，特殊处理
    'dd': None,
    'dt': None,
    'td': None,
}


def _parse_integration(integ_str: str) -> float:
    """将 '3H', '2H' 等字符串转为数字"""
    m = re.search(r'[\d.]+', str(integ_str))
    return float(m.group()) if m else 1.0


def _multiplet_peaks(shift_center: float, multiplicity: str,
                     J_list: list, freq_mhz: float, rng):
    """
    返回 (positions_ppm, relative_heights) 列表。
    支持 s/d/t/q/p/h/hept/m/dd/dt/td。
    """
    mult = multiplicity.lower().strip()

    # 解析耦合常数（Hz → ppm）
    J_ppm = [float(re.search(r'[\d.]+', str(j)).group()) / freq_mhz
             for j in J_list if re.search(r'[\d.]+', str(j))]

    if mult == 's':
        return [shift_center], [1.0]

    if mult in ('d', 't', 'q', 'p', 'h', 'hept'):
        offsets_n, heights = _MULTIPLET_PATTERNS[mult]
        J = J_ppm[0] if J_ppm else 0.01
        positions = [shift_center + o * J for o in offsets_n]
        return positions, [float(h) for h in heights]

    if mult == 'dd':
        # 两个 d，逐步裂分
        J1 = J_ppm[0] if len(J_ppm) > 0 else 0.01
        J2 = J_ppm[1] if len(J_ppm) > 1 else J1 * 0.5
        positions = []
        for o1 in [-J1, J1]:
            for o2 in [-J2, J2]:
                positions.append(shift_center + (o1 + o2) / 2)
        return sorted(positions), [1.0] * 4

    if mult in ('dt', 'td'):
        J1 = J_ppm[0] if len(J_ppm) > 0 else 0.01
        J2 = J_ppm[1] if len(J_ppm) > 1 else J1 * 0.5
        if mult == 'dt':
            outer_off, outer_h = [-J1, J1], [1, 1]
            inner_off, inner_h = [-J2, 0, J2], [1, 2, 1]
        else:
            outer_off, outer_h = [-J1, 0, J1], [1, 2, 1]
            inner_off, inner_h = [-J2, J2], [1, 1]
        positions, heights = [], []
        for oo, oh in zip(outer_off, outer_h):
            for io, ih in zip(inner_off, inner_h):
                positions.append(shift_center + oo / 2 + io / 2)
                heights.append(oh * ih)
        return positions, [float(h) for h in heights]

    # 'm'：宽多重峰，用若干随机小峰模拟
    n_sub = rng.integers(6, 12)
    span = max(J_ppm[0] if J_ppm else 0.02, 0.02)
    sub_pos = rng.uniform(shift_center - span, shift_center + span, n_sub)
    sub_h = rng.uniform(0.5, 1.0, n_sub)
    return sub_pos.tolist(), sub_h.tolist()


def HygrogenToSpectra(data: dict, image_path: str = 'spectra_1H.png') -> plt.Figure:
    """
    将 1H NMR 离散数据转化为伪实验谱图并保存。

    Parameters
    ----------
    data : dict
        包含 'frequency', 'solvent', 'data' 键的字典。
        'data' 为字符串，表示
        [(multiplicity, [J_couplings], integration, shift_max, shift_center), ...] 列表。
    image_path : str
        输出图片路径。

    Returns
    -------
    matplotlib.figure.Figure
    """
    rng = np.random.default_rng(2025)

    # ── 解析数据 ──
    raw = _parse_data_string(data['data'])
    freq_mhz = float(re.search(r'[\d.]+', data['frequency']).group())
    freq_label = data['frequency']

    # ── 谱图参数 ──
    ppm_min, ppm_max = 0, 12
    x = np.linspace(ppm_min, ppm_max, 200000)
    y = np.zeros_like(x)

    lw_base = 0.01    # ppm，1H 线型较窄
    eta = 0.55

    # 收集各峰的积分权重，用于后续绘制积分曲线
    integral_data = []   # [(shift_center, n_H)]

    for entry in raw:
        mult, J_list, integ_str, shift_max, shift_center = entry
        n_H = _parse_integration(integ_str)

        positions, heights = _multiplet_peaks(
            shift_center, mult, J_list, freq_mhz, rng)

        lw = lw_base * rng.uniform(0.9, 1.1)
        h_sum = sum(heights)
        for pos, h in zip(positions, heights):
            amplitude = n_H * h / h_sum
            y += amplitude * _pseudo_voigt(x, pos, lw, eta)

        integral_data.append((shift_center, shift_max, n_H))

    # 归一化 + 噪声
    y_max = y.max()
    y /= y_max
    y = _add_noise(y, snr=500.0, rng=rng)
    y = np.clip(y, -0.03, None)

    # ── 双子图布局：上谱线，下积分 ──
    # 参考图风格：谱线与积分曲线在各自独立的行，积分行高约为谱线行的 1/4
    fig, (ax_spec, ax_int) = plt.subplots(
        2, 1, figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI,
        gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.0}
    )



    # ── 谱线子图 ──
    ax_spec.plot(x, y, color='black', linewidth=0.8, zorder=3)
    ax_spec.set_xlim(ppm_max, ppm_min)
    ax_spec.spines['top'].set_visible(False)
    ax_spec.spines['right'].set_visible(False)
    ax_spec.spines['bottom'].set_visible(False)
    ax_spec.tick_params(axis='x', bottom=False, labelbottom=False)
    ax_spec.tick_params(axis='y', left=False, labelleft=False)
    
    # ── 积分子图（只显示数字）──
    ax_int.set_xlim(ppm_max, ppm_min)
    ax_int.set_ylim(0, 1)
    ax_int.spines['top'].set_visible(False)
    ax_int.spines['right'].set_visible(False)
    ax_int.spines['left'].set_visible(False)
    ax_int.set_xlabel("Chemical Shift (ppm)", fontsize=11, labelpad=6)
    ax_int.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax_int.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax_int.tick_params(axis='x', direction='out', length=4)

    # 只显示数值，竖排居中
    for shift_center, shift_max, n_H in integral_data:
        x_mid = (shift_center + shift_max) / 2
        label = f'{n_H:.2f}' if n_H != int(n_H) else f'{int(n_H):.2f}'
        ax_int.text(x_mid, 0.5, label,
                    ha='center', va='center', fontsize=8,
                    color='black', rotation=90,
                    fontfamily='monospace')

    # 共享 x 轴刻度
    ax_spec.set_xlim(ppm_max, ppm_min)
    ax_int.set_xlim(ppm_max, ppm_min)

    plt.tight_layout()
    fig.savefig(image_path, dpi=DPI, bbox_inches='tight')
    print(f"[1H] 谱图已保存至 {image_path}")
    return fig


# ─────────────────────────────────────────────────────────────────
#  统一入口
# ─────────────────────────────────────────────────────────────────

def SpectraToImage(spectra: dict, type: str, image_path: str = 'spectra_image.png') -> plt.Figure:
    """
    将 NMR 离散数据转化为谱图。

    Parameters
    ----------
    spectra : dict
        NMR 数据字典，包含 'frequency', 'solvent', 'data' 键。
    type : str
        '1H' 或 '13C'。
    image_path : str
        输出图片路径。

    Returns
    -------
    matplotlib.figure.Figure
    """
    if type == '1H':
        return HygrogenToSpectra(spectra, image_path)
    elif type == '13C':
        return CarbonToSpectra(spectra, image_path)
    else:
        raise ValueError(f"Unknown NMR type: {type}. Expected '1H' or '13C'.")


# ─────────────────────────────────────────────────────────────────
#  演示入口
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    data_13C = {
        'frequency': '101 MHz',
        'solvent': 'CDCl3',
        'data': (
            "[(180.0, 'd', 300.8), (167.6, 'd', 22.7), (137.1, 'd', 2.2), "
            "(133.7, None, None), (132.2, None, None), (129.7, 'd', 13.2), "
            "(128.0, None, None), (127.7, 'd', 2.2), (127.4, None, None), "
            "(126.9, None, None), (126.0, None, None), (125.4, None, None), "
            "(60.8, None, None), (31.3, 'd', 11.4), (14.3, None, None), "
            "(7.7, None, None), (3.0, 'd', 3.7)]"
        )
    }

    data_1H = {
        'frequency': '400 MHz',
        'solvent': 'CDCl3',
        'data': (
            "[('m', [], '3H', 7.8, 7.74), ('s', [], '1H', 7.64, 7.64), "
            "('m', [], '3H', 7.45, 7.35), ('q', ['7.1Hz'], '2H', 4.13, 4.13), "
            "('d', ['4.2Hz'], '2H', 3.89, 3.89), ('t', ['7.2Hz'], '3H', 1.21, 1.21), "
            "('t', ['7.8Hz'], '9H', 1.0, 1.0), ('q', ['7.4Hz'], '6H', 0.83, 0.83)]"
        )
    }

    fig_c = SpectraToImage(data_13C, '13C', 'spectra_13C.png')
    fig_h = SpectraToImage(data_1H,  '1H',  'spectra_1H.png')
    plt.show()