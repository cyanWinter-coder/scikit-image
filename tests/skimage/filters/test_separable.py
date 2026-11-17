import numpy as np
from numpy.testing import assert_allclose
from scipy import ndimage as ndi

from skimage.filters import separable_convolve, separable_filter


def test_separable_filter_public_api():
    rng = np.random.default_rng(87235)
    image = rng.normal(size=(5, 6, 7)).astype(np.float32)

    result = separable_filter(
        image,
        sigma=(0.8, 1.1, 0.6),
        spacing=(1.0, 2.0, 0.5),
        mode="mirror",
        chunk_size=2,
    )
    expected = ndi.gaussian_filter(image, sigma=(0.8, 0.55, 1.2), mode="mirror")
    assert result.dtype == np.float32
    assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_separable_convolve_public_api():
    image = np.arange(4 * 5 * 6, dtype=float).reshape((4, 5, 6))
    kernels = ([0.25, 0.5, 0.25],) * 3
    result = separable_convolve(image, kernels, mode="reflect")
    expected = ndi.convolve(
        image,
        np.outer(kernels[0], np.outer(kernels[1], kernels[2])).reshape(3, 3, 3),
        mode="reflect",
    )
    assert_allclose(result, expected)
