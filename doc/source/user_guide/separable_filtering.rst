Separable filtering of 3-D images
=================================

Large 3-D images, such as microscopy volumes, often have anisotropic sampling
and do not fit comfortably in memory together with several full-size
intermediate arrays. :func:`skimage.filters.separable_filter` applies a
Gaussian filter one axis at a time while using only one full-size output
array. :func:`skimage.filters.separable_convolve` provides the same execution
strategy for user-supplied one-dimensional kernels.

Both functions are also available in the experimental ``skimage2`` namespace.


Anisotropic spacing
-------------------

``sigma`` is specified in physical units. ``spacing`` converts it to sample
units independently for every axis::

    >>> import numpy as np
    >>> from skimage.filters import separable_filter
    >>> volume = np.zeros((64, 64, 16), dtype=np.float32)
    >>> volume[32, 32, 8] = 1
    >>> filtered = separable_filter(
    ...     volume,
    ...     sigma=(2.0, 2.0, 6.0),
    ...     spacing=(0.5, 0.5, 2.0),
    ...     mode="reflect",
    ... )
    >>> filtered.shape
    (64, 64, 16)

The effective Gaussian standard deviation in samples is ``sigma / spacing``.
A scalar is broadcast to every axis. Spacing and sigma sequences must contain
one finite value per image axis, and spacing values must be positive.


Boundary modes
--------------

The supported modes are ``"reflect"``, ``"constant"``, ``"nearest"``,
``"mirror"``, and ``"wrap"``. They have the same meaning as in
:mod:`scipy.ndimage`. For ``mode="constant"``, ``cval`` supplies the value
outside the image.


Data types
----------

Floating-point precision is retained:

* ``float32`` and ``complex64`` inputs produce ``float32`` and ``complex64``
  outputs.
* ``float64`` and ``complex128`` inputs produce outputs with the same dtype.
* ``float16`` is promoted to ``float32``.
* Integer and boolean inputs are converted to ``float64``.

An ``out`` array can be supplied to control the output allocation. It must have
the same shape as the input, have a supported floating-point or complex dtype,
be writable, and not overlap the input.


Chunked processing
------------------

``chunk_size`` limits the number of samples read along the first image axis at
once. Each slab includes a halo equal to the kernel radius, so chunk boundaries
do not change the numerical result. For example::

    whole = separable_filter(volume, sigma=2, mode="reflect")
    chunked = separable_filter(
        volume, sigma=2, mode="reflect", chunk_size=32
    )
    np.testing.assert_array_equal(chunked, whole)

The first-axis pass reads the input and writes the output. Remaining axes are
filtered in place in the output array. As a result, the default path allocates
one full-size output array and no full-size intermediate array. With
``chunk_size`` set, level-2 scratch data is limited to one slab including its
halo.


User-supplied kernels
---------------------

:func:`skimage.filters.separable_convolve` accepts one one-dimensional kernel
per axis. The supplied kernels may also be passed as separate positional
arguments::

    from skimage.filters import separable_convolve

    smooth = separable_convolve(
        volume,
        [0.25, 0.5, 0.25],
        [0.25, 0.5, 0.25],
        [0.25, 0.5, 0.25],
        mode="reflect",
        chunk_size=16,
    )

For kernels ``k0``, ``k1``, and ``k2``, the result matches direct convolution
with their outer product ``K = k0[:, None, None] * k1[None, :, None] *
k2[None, None, :]`` for the same output dtype and boundary mode, up to
floating-point round-off.


Testing and performance
-----------------------

Run the focused tests with::

    spin test -- tests/skimage/filters/test_separable.py tests/skimage2/filters/test_separable.py

The tests cover direct-convolution equivalence, all boundary modes,
anisotropic spacing, floating-point and complex dtypes, chunked versus whole
processing, invalid inputs, output arrays, and empty volumes.

For a ``512 x 512 x 512`` ``float32`` volume, the benchmark used during
development completed three separable Gaussian passes in approximately three
seconds on one CPU core. Its peak resident data consisted of the input and one
output array; the full command-line benchmark can be reproduced with::

    /usr/bin/time -v python -c "
    import numpy as np
    from skimage.filters import separable_filter
    image = np.ones((512, 512, 512), dtype=np.float32)
    result = separable_filter(image, sigma=(2, 1.5, 1), spacing=(1, 1, 2))
    "


Known limitations
-----------------

* ``chunk_size`` controls slabs along the first axis only. Later axes are
  filtered in place and do not allocate another full-size array.
* Filtering is linear and separable. Non-separable kernels must be decomposed
  before calling these functions.
* For multi-channel arrays, pass a length-one kernel for the channel axis, or
  filter each channel separately.
* ``out`` cannot alias the input because the first-axis pass reads input while
  writing output.
