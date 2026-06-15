"""Tests for spectrum rendering."""

from PIL import Image

from spectralm.spectra.render import COMBINED_HEIGHT_PX, WIDTH_PX, combine_spectra


def test_combine_spectra_returns_stable_rgb_image(ethanol_sample) -> None:
    """Combined rendering should return an RGB PIL image with stable dimensions."""
    image = combine_spectra(ethanol_sample, h_snr=500.0, c_snr=500.0)
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (WIDTH_PX, COMBINED_HEIGHT_PX)

