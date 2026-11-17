"""Separable multi-dimensional filtering with bounded scratch memory."""

from __future__ import annotations

import operator

import numpy as np
from scipy import ndimage as ndi

from .._shared.utils import _supported_float_type


_ALLOWED_MODES = frozenset({"reflect", "constant", "nearest", "mirror", "wrap"})


def _gaussian_kernel1d(sigma, truncate):
    """Return a normalized 1-D Gaussian kernel sampled on a unit grid."""
    if sigma == 0:
        return np.ones(1, dtype=np.float64)

    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def _validate_mode(mode, function_name):
    if not isinstance(mode, str) or mode not in _ALLOWED_MODES:
        allowed = ", ".join(repr(name) for name in sorted(_ALLOWED_MODES))
        raise ValueError(
            f"{function_name}: `mode` must be one of {allowed}; got {mode!r}."
        )


def _validate_chunk_size(chunk_size, function_name):
    if chunk_size is None:
        return None
    try:
        chunk_size = operator.index(chunk_size)
    except TypeError as exc:
        raise TypeError(
            f"{function_name}: `chunk_size` must be a positive integer or None; "
            f"got {chunk_size!r}."
        ) from exc
    if chunk_size <= 0:
        raise ValueError(
            f"{function_name}: `chunk_size` must be greater than zero; "
            f"got {chunk_size!r}."
        )
    return chunk_size


def _validate_truncate(truncate, function_name):
    try:
        truncate = float(truncate)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{function_name}: `truncate` must be a non-negative finite scalar; "
            f"got {truncate!r}."
        ) from exc
    if not np.isfinite(truncate) or truncate < 0:
        raise ValueError(
            f"{function_name}: `truncate` must be a non-negative finite scalar; "
            f"got {truncate!r}."
        )
    return truncate


def _validate_cval(cval, function_name):
    try:
        value = np.asarray(cval)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{function_name}: `cval` must be a numeric scalar; got {cval!r}."
        ) from exc
    if value.ndim != 0 or value.dtype.kind not in "biufc":
        raise TypeError(
            f"{function_name}: `cval` must be a numeric scalar; got {cval!r}."
        )


def _normalize_axis_values(value, ndim, name, *, function_name, positive):
    try:
        values = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{function_name}: `{name}` must be a scalar or a sequence of "
            f"{ndim} numeric values; got {value!r}."
        ) from exc

    if values.ndim == 0:
        values = np.full(ndim, values.item(), dtype=np.float64)
    elif values.ndim != 1 or values.size != ndim:
        raise ValueError(
            f"{function_name}: `{name}` must be a scalar or a sequence with "
            f"one value per image axis ({ndim}); got shape {values.shape}."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{function_name}: `{name}` must contain only finite values; "
            f"got {values!r}."
        )
    if positive and np.any(values <= 0):
        raise ValueError(
            f"{function_name}: `{name}` must contain only positive values; "
            f"got {values!r}."
        )
    if not positive and np.any(values < 0):
        raise ValueError(
            f"{function_name}: `{name}` must contain only non-negative values; "
            f"got {values!r}."
        )
    return values


def _normalize_kernels(kernels, ndim, function_name):
    try:
        kernels = tuple(kernels)
    except TypeError as exc:
        raise TypeError(
            f"{function_name}: `kernels` must be a sequence containing one "
            f"1-D kernel per image axis ({ndim}); got {kernels!r}."
        ) from exc

    if len(kernels) != ndim:
        raise ValueError(
            f"{function_name}: `kernels` must contain one 1-D kernel per image "
            f"axis ({ndim}); got {len(kernels)} kernels."
        )

    normalized = []
    for axis, kernel in enumerate(kernels):
        try:
            kernel = np.asarray(kernel, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{function_name}: kernel for axis {axis} must be a numeric "
                f"1-D array; got {kernel!r}."
            ) from exc
        if kernel.ndim != 1:
            raise ValueError(
                f"{function_name}: kernel for axis {axis} must be one "
                f"dimensional; got shape {kernel.shape}."
            )
        if kernel.size == 0:
            raise ValueError(
                f"{function_name}: kernel for axis {axis} must not be empty."
            )
        if not np.all(np.isfinite(kernel)):
            raise ValueError(
                f"{function_name}: kernel for axis {axis} must contain only "
                "finite values."
            )
        normalized.append(kernel)
    return tuple(normalized)


def _prepare_output(image, out, function_name):
    if out is None:
        output_dtype = _supported_float_type(image.dtype, allow_complex=True)
        try:
            return np.empty(image.shape, dtype=output_dtype)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{function_name}: image data cannot be represented as "
                f"{output_dtype}; got dtype {image.dtype}."
            ) from exc

    if not isinstance(out, np.ndarray):
        raise TypeError(
            f"{function_name}: `out` must be a NumPy ndarray; got {type(out)!r}."
        )
    if out.shape != image.shape:
        raise ValueError(
            f"{function_name}: `out` must have the same shape as `image` "
            f"{image.shape}; got {out.shape}."
        )
    if not np.issubdtype(out.dtype, np.inexact):
        raise ValueError(
            f"{function_name}: dtype of `out` must be floating-point or complex; "
            f"got {out.dtype!r}."
        )
    _validate_ndimage_dtype(out.dtype, "out", function_name)
    if not out.flags.writeable:
        raise ValueError(f"{function_name}: `out` must be writable.")
    if np.may_share_memory(image, out):
        raise ValueError(
            f"{function_name}: `out` must not overlap `image`; filtering writes "
            "the result while reading the input."
        )
    if np.iscomplexobj(image) and not np.iscomplexobj(out):
        raise ValueError(
            f"{function_name}: dtype of `out` must be complex for complex input; "
            f"got {out.dtype!r}."
        )
    return out


def _axis_slice(array, axis, start, stop):
    index = [slice(None)] * array.ndim
    index[axis] = slice(start, stop)
    return array[tuple(index)]


def _copy_to_output(image, output):
    try:
        np.copyto(output, image)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "separable filtering: image data cannot be converted to the output "
            f"dtype {output.dtype}; got dtype {image.dtype}."
        ) from exc


def _filter_axis_from_source(source, output, kernel, axis, mode, cval, chunk_size):
    """Filter one axis from `source` into `output`, optionally in slabs."""
    length = source.shape[axis]
    radius = kernel.size // 2

    if chunk_size is None or radius == 0 or chunk_size >= length:
        ndi.convolve1d(
            source,
            kernel,
            axis=axis,
            output=output,
            mode=mode,
            cval=cval,
        )
        return

    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        if mode == "wrap":
            lower = start - radius
            upper = stop + radius
            indices = np.arange(lower, upper) % length
            source_slab = np.take(source, indices, axis=axis)
            slab_mode = "nearest"
            central_start = radius
        else:
            lower = max(0, start - radius)
            upper = min(length, stop + radius)
            source_slab = _axis_slice(source, axis, lower, upper)
            slab_mode = mode
            central_start = start - lower

        filtered_slab = ndi.convolve1d(
            source_slab,
            kernel,
            axis=axis,
            mode=slab_mode,
            cval=cval,
        )
        central_stop = central_start + (stop - start)
        _axis_slice(output, axis, start, stop)[...] = _axis_slice(
            filtered_slab, axis, central_start, central_stop
        )


def _filter_inplace(output, kernel, axis, mode, cval):
    ndi.convolve1d(
        output,
        kernel,
        axis=axis,
        output=output,
        mode=mode,
        cval=cval,
    )


def _apply_separable_filters(
    image,
    output,
    kernels,
    active_axes,
    mode,
    cval,
    chunk_size,
):
    if not active_axes:
        _copy_to_output(image, output)
        return output

    first_axis = active_axes[0]
    axis_cval = cval
    if first_axis == 0:
        _filter_axis_from_source(
            image,
            output,
            kernels[0],
            axis=0,
            mode=mode,
            cval=axis_cval,
            chunk_size=chunk_size,
        )
        if mode == "constant":
            axis_cval *= kernels[0].sum()
        remaining_axes = active_axes[1:]
    else:
        _copy_to_output(image, output)
        remaining_axes = active_axes

    for axis in remaining_axes:
        _filter_inplace(output, kernels[axis], axis, mode, axis_cval)
        if mode == "constant":
            axis_cval *= kernels[axis].sum()
    return output


def _validate_ndimage_dtype(dtype, name, function_name):
    dtype = np.dtype(dtype)
    supported = (
        dtype.kind in "biu"
        or (dtype.kind == "f" and dtype.itemsize in (4, 8))
        or (dtype.kind == "c" and dtype.itemsize in (8, 16))
    )
    if not supported:
        raise TypeError(
            f"{function_name}: `{name}` has unsupported dtype {dtype!r}; "
            "SciPy filtering supports integer, float32, float64, complex64, "
            "and complex128 arrays."
        )


def _validate_image_dtype(image, function_name):
    dtype = image.dtype
    if dtype.kind not in "biufc":
        raise TypeError(
            f"{function_name}: `image` must have a numeric dtype; got {dtype!r}."
        )


def _ndimage_source(image, output_dtype):
    dtype = image.dtype
    input_supported = (
        dtype.kind in "biu"
        or (dtype.kind == "f" and dtype.itemsize in (4, 8))
        or (dtype.kind == "c" and dtype.itemsize in (8, 16))
    )
    if input_supported:
        return image
    return np.asarray(image, dtype=output_dtype)


def separable_filter(
    image,
    sigma=1.0,
    *,
    spacing=None,
    mode="nearest",
    cval=0.0,
    truncate=4.0,
    chunk_size=None,
    out=None,
):
    """Filter an image with a separable Gaussian kernel.

    The filter is applied one axis at a time. This is optimized for 3-D
    volumes, but also works for arrays with other dimensionalities. Physical
    standard deviations are converted to sample units using ``spacing``,
    which makes the result independent of voxel anisotropy.

    Parameters
    ----------
    image : array-like
        Input image. Floating-point precision is preserved: ``float32`` and
        ``complex64`` inputs produce ``float32`` and ``complex64`` outputs,
        respectively. ``float16`` is promoted to ``float32`` and integer input
        is converted to ``float64``.
    sigma : scalar or sequence of scalars, optional
        Gaussian standard deviation in physical units. A scalar is used for
        every axis. A sequence must contain one non-negative value per image
        axis.
    spacing : scalar or sequence of scalars, optional
        Physical spacing between samples along each image axis. A scalar is
        used for every axis. The Gaussian kernel width in samples is
        ``sigma / spacing``. By default, unit spacing is assumed.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        How the image is extended beyond its boundaries. ``cval`` is used only
        when ``mode='constant'``.
    cval : scalar, optional
        Value used outside the image for ``mode='constant'``.
    truncate : float, optional
        Truncate the Gaussian kernel at this many standard deviations.
    chunk_size : int or None, optional
        Number of samples along the first image axis processed at once. Slabs
        include the required kernel halo, so chunked and whole-volume results
        are identical. ``None`` processes the first axis in one pass.
    out : ndarray, optional
        Output array. It must have the same shape as ``image``, have a
        floating-point or complex dtype, be writable, and must not overlap
        ``image``.

    Returns
    -------
    filtered_image : ndarray
        The filtered image.

    Notes
    -----
    Filtering is performed in two stages. The first axis is read from the
    input and written to the output, optionally one slab at a time. Remaining
    axes are filtered in place in the output array. Consequently, at most one
    full-size output array is allocated; level-2 intermediate data is limited
    to the current slab when ``chunk_size`` is set.

    Examples
    --------
    >>> import numpy as np
    >>> import _skimage2 as ski2
    >>> volume = np.zeros((5, 5, 5))
    >>> volume[2, 2, 2] = 1
    >>> filtered = ski2.filters.separable_filter(
    ...     volume, sigma=2, spacing=(1, 1, 4), mode='reflect'
    ... )
    >>> filtered.shape
    (5, 5, 5)
    >>> bool(np.isclose(filtered.sum(), 1.0))
    True
    """
    function_name = "separable_filter"
    try:
        image = np.asarray(image)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{function_name}: `image` cannot be converted to an array; "
            f"got {type(image)!r}."
        ) from exc
    if image.ndim == 0:
        raise ValueError(
            f"{function_name}: `image` must have at least one dimension; "
            f"got shape {image.shape}."
        )
    _validate_image_dtype(image, function_name)

    _validate_mode(mode, function_name)
    sigma = _normalize_axis_values(
        sigma,
        image.ndim,
        "sigma",
        function_name=function_name,
        positive=False,
    )
    if spacing is None:
        spacing = np.ones(image.ndim, dtype=np.float64)
    else:
        spacing = _normalize_axis_values(
            spacing,
            image.ndim,
            "spacing",
            function_name=function_name,
            positive=True,
        )
    truncate = _validate_truncate(truncate, function_name)
    _validate_cval(cval, function_name)
    chunk_size = _validate_chunk_size(chunk_size, function_name)
    output = _prepare_output(image, out, function_name)

    if image.size == 0:
        return output

    output_dtype = output.dtype
    source = _ndimage_source(image, output_dtype)
    sigma_samples = sigma / spacing
    kernels = tuple(_gaussian_kernel1d(value, truncate) for value in sigma_samples)
    active_axes = tuple(axis for axis, value in enumerate(sigma) if value != 0)
    return _apply_separable_filters(
        source,
        output,
        kernels,
        active_axes,
        mode,
        cval,
        chunk_size,
    )


def separable_convolve(
    image,
    kernels,
    *more_kernels,
    mode="nearest",
    cval=0.0,
    chunk_size=None,
    out=None,
):
    """Convolve an image with one kernel along each axis.

    Parameters
    ----------
    image : array-like
        Input image. Floating-point precision is preserved: ``float32`` and
        ``complex64`` inputs produce ``float32`` and ``complex64`` outputs,
        respectively. ``float16`` is promoted to ``float32`` and integer input
        is converted to ``float64``.
    kernels : sequence of array-like
        One non-empty, finite, one-dimensional kernel per image axis. Kernels
        may alternatively be passed as separate positional arguments, e.g.
        ``separable_convolve(image, k0, k1, k2)``.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        How the image is extended beyond its boundaries.
    cval : scalar, optional
        Value used outside the image for ``mode='constant'``.
    chunk_size : int or None, optional
        Number of samples along the first image axis processed at once. Slabs
        include the required kernel halo.
    out : ndarray, optional
        Output array. It must have the same shape as ``image``, have a
        floating-point or complex dtype, be writable, and must not overlap
        ``image``.

    Returns
    -------
    filtered_image : ndarray
        The filtered image.

    Notes
    -----
    For a full kernel ``K`` equal to the outer product of the supplied
    one-dimensional kernels, the result matches
    ``scipy.ndimage.convolve(image, K, mode=mode, cval=cval)`` for the same
    output dtype, up to floating-point round-off.

    Examples
    --------
    >>> import numpy as np
    >>> import _skimage2 as ski2
    >>> volume = np.zeros((7, 7, 7))
    >>> volume[3, 3, 3] = 1
    >>> smooth = ski2.filters.separable_convolve(
    ...     volume, [0.25, 0.5, 0.25], [0.25, 0.5, 0.25], [0.25, 0.5, 0.25]
    ... )
    >>> smooth.shape
    (7, 7, 7)
    """
    function_name = "separable_convolve"
    try:
        image = np.asarray(image)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{function_name}: `image` cannot be converted to an array; "
            f"got {type(image)!r}."
        ) from exc
    if image.ndim == 0:
        raise ValueError(
            f"{function_name}: `image` must have at least one dimension; "
            f"got shape {image.shape}."
        )
    _validate_image_dtype(image, function_name)

    if more_kernels:
        kernels = (kernels, *more_kernels)
    _validate_mode(mode, function_name)
    _validate_cval(cval, function_name)
    kernels = _normalize_kernels(kernels, image.ndim, function_name)
    chunk_size = _validate_chunk_size(chunk_size, function_name)
    output = _prepare_output(image, out, function_name)

    if image.size == 0:
        return output

    output_dtype = output.dtype
    source = _ndimage_source(image, output_dtype)
    active_axes = tuple(range(image.ndim))
    return _apply_separable_filters(
        source,
        output,
        kernels,
        active_axes,
        mode,
        cval,
        chunk_size,
    )


__all__ = ["separable_convolve", "separable_filter"]
