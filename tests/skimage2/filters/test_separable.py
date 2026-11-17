import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import ndimage as ndi

from _skimage2._shared.utils import _supported_float_type
from _skimage2.filters import separable_convolve, separable_filter


def _gaussian_kernels(sigma, truncate):
    kernels = []
    for value in sigma:
        if value == 0:
            kernels.append(np.ones(1))
            continue
        radius = int(truncate * value + 0.5)
        x = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (x / value) ** 2)
        kernels.append(kernel / kernel.sum())
    return kernels


@pytest.mark.parametrize("shape", [(7, 8, 5), (2, 5, 9), (1, 6, 4)])
@pytest.mark.parametrize(
    "sigma",
    [
        1.1,
        (0.6, 1.2, 0.9),
        (0.0, 1.3, 0.7),
    ],
)
@pytest.mark.parametrize("spacing", [None, (1.0, 2.0, 0.5)])
@pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
def test_separable_filter_matches_direct_convolution(shape, sigma, spacing, mode):
    rng = np.random.default_rng(546834)
    image = rng.normal(size=shape)
    sigma_samples = np.broadcast_to(np.asarray(sigma, dtype=float), (3,))
    if spacing is not None:
        sigma_samples = sigma_samples / np.asarray(spacing)

    kernels = _gaussian_kernels(sigma_samples, truncate=3.5)
    full_kernel = (
        kernels[0][:, np.newaxis, np.newaxis]
        * kernels[1][np.newaxis, :, np.newaxis]
        * kernels[2][np.newaxis, np.newaxis, :]
    )
    expected = ndi.convolve(image, full_kernel, mode=mode, cval=0.3)
    result = separable_filter(
        image,
        sigma=sigma,
        spacing=spacing,
        mode=mode,
        cval=0.3,
        truncate=3.5,
    )

    assert_allclose(result, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 100])
@pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
def test_separable_filter_chunked_matches_whole(chunk_size, mode):
    rng = np.random.default_rng(932867)
    image = rng.random((9, 7, 6), dtype=np.float32)

    whole = separable_filter(
        image,
        sigma=(1.2, 0.8, 0.5),
        spacing=(1.0, 2.0, 0.75),
        mode=mode,
        cval=0.2,
    )
    chunked = separable_filter(
        image,
        sigma=(1.2, 0.8, 0.5),
        spacing=(1.0, 2.0, 0.75),
        mode=mode,
        cval=0.2,
        chunk_size=chunk_size,
    )

    assert_array_equal(chunked, whole)


@pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
def test_separable_convolve_matches_direct_convolution(mode):
    rng = np.random.default_rng(18459)
    image = rng.normal(size=(7, 8, 5))
    kernels = (
        np.array([0.1, 0.2, 0.4, 0.2, 0.1]),
        np.array([0.3, 0.4, 0.3]),
        np.array([0.25, 0.5, 0.25]),
    )
    full_kernel = (
        kernels[0][:, np.newaxis, np.newaxis]
        * kernels[1][np.newaxis, :, np.newaxis]
        * kernels[2][np.newaxis, np.newaxis, :]
    )
    expected = ndi.convolve(image, full_kernel, mode=mode, cval=0.4)
    result = separable_convolve(image, *kernels, mode=mode, cval=0.4, chunk_size=2)

    assert_allclose(result, expected, rtol=1e-14, atol=1e-14)


def test_separable_convolve_unnormalized_kernels_constant_mode():
    rng = np.random.default_rng(83471)
    image = rng.normal(size=(8, 7, 6))
    kernels = tuple(rng.normal(size=size) for size in (4, 5, 2))
    full_kernel = (
        kernels[0][:, np.newaxis, np.newaxis]
        * kernels[1][np.newaxis, :, np.newaxis]
        * kernels[2][np.newaxis, np.newaxis, :]
    )
    expected = ndi.convolve(image, full_kernel, mode="constant", cval=0.3)

    for chunk_size in (None, 1, 2, 3):
        result = separable_convolve(
            image,
            kernels,
            mode="constant",
            cval=0.3,
            chunk_size=chunk_size,
        )
        assert_allclose(result, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize(
    "dtype,expected_dtype",
    [
        (np.uint8, np.float64),
        (np.int32, np.float64),
        (np.float16, np.float32),
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.complex64, np.complex64),
        (np.complex128, np.complex128),
    ],
)
def test_separable_filter_dtype(dtype, expected_dtype):
    image = np.arange(4 * 5 * 6).reshape((4, 5, 6)).astype(dtype)
    result = separable_filter(image, sigma=0.8)
    assert result.dtype == expected_dtype
    assert result.dtype == _supported_float_type(image.dtype, allow_complex=True)


def test_separable_filter_anisotropic_spacing():
    image = np.zeros((7, 7, 7))
    image[3, 3, 3] = 1

    isotropic = separable_filter(image, sigma=1.5)
    anisotropic = separable_filter(image, sigma=1.5, spacing=(1, 1, 3))

    assert not np.allclose(isotropic, anisotropic)
    expected = ndi.gaussian_filter(image, sigma=(1.5, 1.5, 0.5), mode="nearest")
    assert_allclose(anisotropic, expected, rtol=1e-13, atol=1e-13)


def test_separable_filter_zero_sigma_is_identity():
    rng = np.random.default_rng(7234)
    image = rng.normal(size=(4, 5, 6)).astype(np.float32)
    result = separable_filter(image, sigma=0)
    assert result.dtype == image.dtype
    assert_array_equal(result, image)


@pytest.mark.parametrize("shape", [(0, 4, 5), (4, 0, 5), (4, 5, 0), (0, 0, 0)])
def test_separable_filter_empty_image(shape):
    image = np.empty(shape, dtype=np.float32)
    result = separable_filter(image, sigma=(1, 2, 3), chunk_size=2)
    assert result.shape == shape
    assert result.dtype == np.float32
    assert result.size == 0


def test_separable_filter_out():
    rng = np.random.default_rng(12983)
    image = rng.normal(size=(5, 6, 7)).astype(np.float32)
    out = np.empty_like(image)
    result = separable_filter(image, sigma=0.9, out=out)
    assert result is out
    assert_allclose(result, separable_filter(image, sigma=0.9))


def test_separable_filter_does_not_modify_input():
    rng = np.random.default_rng(9283)
    image = rng.normal(size=(5, 6, 7))
    original = image.copy()
    separable_filter(image, sigma=1.2)
    assert_array_equal(image, original)


def test_separable_filter_errors():
    image = np.ones((5, 6, 7))

    with pytest.raises(ValueError, match="`sigma`.*non-negative"):
        separable_filter(image, sigma=(1, -1, 1))
    with pytest.raises(ValueError, match="`sigma`.*one value per image axis"):
        separable_filter(image, sigma=(1, 1))
    with pytest.raises(ValueError, match="`spacing`.*positive"):
        separable_filter(image, sigma=1, spacing=(1, 0, 1))
    with pytest.raises(ValueError, match="`spacing`.*finite"):
        separable_filter(image, sigma=1, spacing=(1, np.nan, 1))
    with pytest.raises(ValueError, match="`mode` must be one of"):
        separable_filter(image, sigma=1, mode="invalid")
    with pytest.raises(ValueError, match="`truncate`.*non-negative"):
        separable_filter(image, sigma=1, truncate=-1)
    with pytest.raises(TypeError, match="`chunk_size`.*positive integer"):
        separable_filter(image, sigma=1, chunk_size=1.5)
    with pytest.raises(ValueError, match="`chunk_size`.*greater than zero"):
        separable_filter(image, sigma=1, chunk_size=0)
    with pytest.raises(ValueError, match="`out` must have the same shape"):
        separable_filter(image, sigma=1, out=np.empty((5, 6, 6)))
    with pytest.raises(ValueError, match="dtype of `out`"):
        separable_filter(image, sigma=1, out=np.empty_like(image, dtype=np.int32))
    with pytest.raises(TypeError, match="`out` must be a NumPy ndarray"):
        separable_filter(image, sigma=1, out=[])
    with pytest.raises(ValueError, match="`out` must not overlap"):
        separable_filter(image, sigma=1, out=image)
    with pytest.raises(TypeError, match="`cval` must be a numeric scalar"):
        separable_filter(image, sigma=1, cval=(0, 1))


def test_separable_filter_read_only_out_error():
    image = np.ones((5, 6, 7))
    out = np.empty_like(image)
    out.flags.writeable = False
    with pytest.raises(ValueError, match="`out` must be writable"):
        separable_filter(image, sigma=1, out=out)


def test_separable_convolve_error():
    image = np.ones((5, 6, 7))

    with pytest.raises(ValueError, match="one 1-D kernel per image axis"):
        separable_convolve(image, [1, 1])
    with pytest.raises(ValueError, match="kernel for axis 0 must be one dimensional"):
        separable_convolve(image, np.ones((3, 3)), [1], [1])
    with pytest.raises(ValueError, match="kernel for axis 0 must not be empty"):
        separable_convolve(image, [], [1], [1])
    with pytest.raises(ValueError, match="kernel for axis 0.*finite"):
        separable_convolve(image, [np.nan], [1], [1])


def test_separable_filter_2d_compatibility():
    rng = np.random.default_rng(3498)
    image = rng.normal(size=(8, 9))
    result = separable_filter(image, sigma=(0.7, 1.3), mode="reflect")
    expected = ndi.gaussian_filter(image, sigma=(0.7, 1.3), mode="reflect")
    assert_allclose(result, expected, rtol=1e-13, atol=1e-13)
