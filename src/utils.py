import numpy as np
import re



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


def parse_integration(integ_str: str) -> float:
    """将 '3H', '2H' 等字符串转为数字"""
    m = re.search(r'[\d.]+', str(integ_str))
    return float(m.group()) if m else 1.0
 