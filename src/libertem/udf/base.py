import numpy as np

from libertem.job.base import Task
from libertem.common.buffers import BufferWrapper


def check_cast(fromvar, tovar):
    if not np.can_cast(fromvar.dtype, tovar.dtype, casting='safe'):
        # FIXME exception or warning?
        raise TypeError("Unsafe automatic casting from %s to %s" % (fromvar.dtype, tovar.dtype))


def merge_assign(dest, src):
    for k in dest:
        check_cast(dest[k], src[k])
        dest[k][:] = src[k]


class UDFTask(Task):
    def __init__(self, partition, idx, make_buffers, init, roi, fn):
        super().__init__(partition=partition, idx=idx)
        self._make_buffers = make_buffers
        self._init = init
        self._fn = fn
        self._roi = roi

    def _get_roi(self):
        partition = self.partition
        roi = BufferWrapper(kind="nav", dtype=np.dtype("bool"))
        roi.set_shape_partition(partition)
        roi.allocate()
        roi_data = roi.data
        ds_shape = partition.meta.raw_shape
        if self._roi is None:
            roi_data[:] = 1
        else:
            roi_reshaped = self._roi.reshape(ds_shape.nav)
            roi_data[:] = roi_reshaped[partition.slice.get(nav_only=True)]
        return roi

    def __call__(self):
        roi = self._get_roi()
        partition = self.partition

        result_buffers = self._make_buffers()
        for buf in result_buffers.values():
            buf.set_shape_partition(partition)
            buf.allocate()
        if self._init is not None:
            kwargs = self._init(partition, **{
                k: v.get_view_for_partition(partition)
                for k, v in result_buffers.items()
            })
        else:
            kwargs = {}
        kwargs.update(result_buffers)
        for tile in partition.get_tiles(full_frames=True):
            self._call_for_tile(tile, roi, result_buffers, kwargs)
        return result_buffers, partition

    def _call_for_tile(self, tile, roi, result_buffers, kwargs):
        partition = self.partition
        data = tile.flat_nav
        for frame_idx, frame in enumerate(data):
            roi_value = roi.get_view_for_frame(
                partition=partition,
                tile=tile,
                frame_idx=frame_idx
            )[0]
            if not roi_value:
                continue
            buffer_views = {}
            for k, buf in result_buffers.items():
                buffer_views[k] = buf.get_view_for_frame(
                    partition=partition,
                    tile=tile,
                    frame_idx=frame_idx
                )
            kwargs.update(buffer_views)
            self._fn(frame=frame, **kwargs)


def make_udf_tasks(dataset, fn, init, make_buffers, roi):
    return (
        UDFTask(partition=partition, idx=idx, fn=fn, init=init, make_buffers=make_buffers, roi=roi)
        for idx, partition in enumerate(dataset.get_partitions())
    )
