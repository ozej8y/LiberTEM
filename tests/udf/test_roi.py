import numpy as np

from libertem.common.buffers import BufferWrapper

from utils import MemoryDataSet, _mk_random


def my_buffers():
    return {
        'pixelsum': BufferWrapper(
            kind="nav", dtype="float32"
        )
    }


def my_frame_fn(frame, pixelsum):
    pixelsum[:] = np.sum(frame)


def test_roi_1(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 2, 16, 16),
                            partition_shape=(3, 3, 16, 16), sig_dims=2)

    roi = np.zeros((16, 16), dtype=bool)
    roi[4:12, 4:12] = True

    res = lt_ctx.run_udf(
        dataset=dataset,
        fn=my_frame_fn,
        make_buffers=my_buffers,
        roi=roi,
    )
    assert 'pixelsum' in res
    print(data.shape, res['pixelsum'].data.shape)
    expected = np.sum(data, axis=(2, 3))
    expected[roi == 0] = 0
    assert np.allclose(res['pixelsum'].data, expected)


def test_roi_randomized(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 2, 16, 16),
                            partition_shape=(3, 3, 16, 16), sig_dims=2)

    roi = np.random.choice([True, False], size=(16, 16))

    res = lt_ctx.run_udf(
        dataset=dataset,
        fn=my_frame_fn,
        make_buffers=my_buffers,
        roi=roi,
    )
    assert 'pixelsum' in res
    print(data.shape, res['pixelsum'].data.shape)
    expected = np.sum(data, axis=(2, 3))
    expected[roi == 0] = 0
    assert np.allclose(res['pixelsum'].data, expected)


def test_roi_2_3d_dataset(lt_ctx):
    data = _mk_random(size=(16 * 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(2, 16, 16),
                            partition_shape=(6, 16, 16), sig_dims=2)

    roi = np.zeros((16, 16,), dtype=bool)
    roi[4:12] = True

    res = lt_ctx.run_udf(
        dataset=dataset,
        fn=my_frame_fn,
        make_buffers=my_buffers,
        roi=roi,
    )
    assert 'pixelsum' in res
    print(data.shape, res['pixelsum'].data.shape)
    expected = np.sum(data, axis=(1, 2))
    expected[roi.reshape(-1) == 0] = 0
    assert np.allclose(res['pixelsum'].data, expected)
