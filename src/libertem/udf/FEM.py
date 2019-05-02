import functools

import numpy as np

from libertem.common.buffers import BufferWrapper
from libertem.masks import _make_circular_mask


def make_result_buffers(num_bins):
    return {
        'intensity': BufferWrapper(
            kind="nav", dtype="float32", extra_shape=(num_bins,),
        ),
    }


def init(partition, center, riros):
    w, h = partition.shape.sig[1], partition.shape.sig[0]

    masks = np.zeros((len(riros), h, w), dtype='bool')

    for idx, (ri, ro) in enumerate(riros):
        mask_out = _make_circular_mask(
            center[1], center[0],
            w, h,
            ro
        )
        mask_in = _make_circular_mask(
            center[1], center[0],
            w, h,
            ri
        )
        mask = mask_out & (~mask_in)
        masks[idx] = mask

    kwargs = {
        'masks': masks,
    }
    return kwargs


def masked_std(frame, masks, intensity):
    for idx, mask in enumerate(masks):
        intensity[idx] = np.std(frame[mask])


def run_fem(ctx, dataset, center, rad_min, rad_max, num_bins):
    """
    Return a standard deviation(SD) value for each frame of pixels which belong to ring mask.
    Parameters
    ----------
    ctx: Context
        Context class that contains methods for loading datasets,
        creating jobs on them and running them

    dataset: DataSet
        A dataset with 1- or 2-D scan dimensions and 2-D frame dimensions

    center: tuple
        (y, x) - coordinates of a center of a ring for a masking region of interest to calculate SD

    rad_min: int
        Inner radius of a ring mask

    rad_max: int
        Outer radius of a ring mask

    num_bins : int
        Number of radial bins

    Returns
    -------
    pass_results: dict
        Returns a standard deviation(SD) value for each frame of pixels which belong to ring mask.
        To return 2-D array use pass_results['intensity'].data

    """

    rads = np.arange(rad_min, rad_max, (rad_max - rad_min) / (num_bins + 1))
    riros = list(zip(rads, rads[1:]))

    pass_results = ctx.run_udf(
        dataset=dataset,
        make_buffers=functools.partial(make_result_buffers, num_bins=len(riros)),
        init=functools.partial(init, center=center, riros=riros),
        fn=masked_std,
    )

    return (pass_results)
