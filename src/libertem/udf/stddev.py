import collections

import numpy as np

from libertem.common.buffers import BufferWrapper


VariancePart = collections.namedtuple('VariancePart', ['var', 'sum_im', 'N'])


def batch_buffer():
    """
    Initializes BufferWrapper objects for sum of variances,
    sum of frames, and the number of frames

    Returns
    -------
    A dictionary that maps 'var', 'std', 'mean', 'num_frame', 'sum_frame' to
    the corresponding BufferWrapper objects
    """
    return {
        'var': BufferWrapper(
            kind='sig', dtype='float32'
            ),
        'num_frame': BufferWrapper(
            kind='single', dtype='float32'
            ),
        'sum_frame': BufferWrapper(
            kind='sig', dtype='float32'
            )
    }


def compute_batch(frame, var, sum_frame, num_frame):
    """
    Given a frame, update sum of variances, sum of frames,
    and the number of total frames

    Parameters
    ----------
    frame
        single frame of the data

    var
        Buffer that stores sum of variances of the previous set of frames

    sum_frame
        Buffer that sores sum of frames of the previous set of frames

    num_frame
        Buffer that stores the number of frames used for computation

    """
    if num_frame == 0:
        var[:] = 0

    else:
        p0 = VariancePart(var=var, sum_im=sum_frame, N=num_frame)
        p1 = VariancePart(var=0, sum_im=frame, N=1)
        compute_merge = merge(p0, p1)

        var[:] = compute_merge.var

    sum_frame[:] += frame
    num_frame[:] += 1


def batch_merge(dest, src):
    """
    Given two buffers that contain sum of variances, sum of frames,
    and the number of frames used in each of the partitions, merge the
    partitions and compute the joint sum of variances and sum of frames
    over all frames used

    Parameters
    ----------
    dest
        Partial results that contains sum of variances, sum of frames, and the
        number of frames used over all the frames used

    src
        Partial results that contains sum of variances, sum of frames, and the
        number of frames used over current iteration of partition
    """
    p0 = VariancePart(var=dest['var'][:],
                    sum_im=dest['sum_frame'][:],
                    N=dest['num_frame'][:])
    p1 = VariancePart(var=src['var'][:],
                    sum_im=src['sum_frame'][:],
                    N=src['num_frame'][:])
    compute_merge = merge(p0, p1)

    dest['var'][:] = compute_merge.var
    dest['sum_frame'][:] = compute_merge.sum_im
    dest['num_frame'][:] = compute_merge.N


def merge(p0, p1):
    """
    Given two sets of partitions, with sum of frames
    and sum of variances, compute joint sum of frames
    and sum of variances using one pass algorithm

    Parameters
    ----------
    p0
        Contains information about the first partition, including
        sum of variances, sum of pixels, and number of frames used

    p1
        Contains information about the second partition, including
        sum of variances, sum of pixels, and number of frames used

    Returns
    -------
    VariancePart
        colletions.namedtuple object that contains information about
        the merged partitions, including sum of variances,
        sum of pixels, and number of frames used
    """
    if p0.N == 0:
        return p1
    N = p0.N + p1.N

    # compute mean for each partitions
    mean_A = (p0.sum_im / p0.N)
    mean_B = (p1.sum_im / p1.N)

    # compute mean for joint samples
    delta = mean_B - mean_A
    mean = mean_A + (p1.N * delta) / (p0.N + p1.N)

    # compute sum of images for joint samples
    sum_im_AB = p0.sum_im + p1.sum_im

    # compute sum of variances for joint samples
    delta_P = mean_B - mean
    var_AB = p0.var + p1.var + (p1.N * delta * delta_P)

    return VariancePart(var=var_AB, sum_im=sum_im_AB, N=N)


def run_stddev(ctx, dataset):
    """
    Compute sum of variances and sum of pixels from the given dataset

    One-pass algorithm used in this code is taken from the following paper:
    "Numerically Stable Parallel Computation of (Co-) Variance"
    DOI : https://doi.org/10.1145/3221269.3223036

    Parameters
    ----------
    ctx
        Context class that contains methods for loading datasets, creating jobs on them
        and running them

    dataset
        dataset to work on

    Returns
    -------
    pass_results
        A dictionary of narrays that contains sum of variances, sum of pixels,
        and number of frames used to compute the above statistic

    To retrieve statistic, using the following commands:
    variance : pass_results['var']
    standard deviation : pass_results['std']
    sum of pixels : pass_results['sum_frame']
    mean : pass_results['mean']
    number of frames : pass_results['num_frame']
    """
    pass_results = ctx.run_udf(
        dataset=dataset,
        fn=compute_batch,
        make_buffers=batch_buffer,
        merge=batch_merge,
    )

    pass_results['var'] = pass_results['var'].data/pass_results['num_frame'].data
    pass_results['std'] = np.sqrt(pass_results['var'].data)
    pass_results['mean'] = pass_results['sum_frame'].data/pass_results['num_frame'].data
    pass_results['num_frame'] = pass_results['num_frame'].data
    pass_results['sum_frame'] = pass_results['sum_frame'].data

    return pass_results
