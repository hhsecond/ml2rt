import importlib


def _get_tensortype(shape, dtype):
    # TODO: remove this once this is added to onnxconverter_common or move this there
    from onnxconverter_common import data_types as onnx_dtypes

    if dtype == 'float32':
        onnx_dtype = onnx_dtypes.FloatTensorType(shape)
    elif dtype == 'float64':
        onnx_dtype = onnx_dtypes.DoubleTensorType(shape)
    elif dtype in ('int64', 'uint64'):
        onnx_dtype = onnx_dtypes.Int64TensorType(shape)
    elif dtype in ('int32', 'uint32'): # noqa
        onnx_dtype = onnx_dtypes.Int32TensorType(shape)
    else:
        raise NotImplementedError(f"'{dtype.name}' is not supported by ONNXRuntime. ")
    return onnx_dtype


def guess_onnx_tensortype(prototype=None, shape=None, dtype=None, node_name='features'):
    # TODO: perhaps move this to onnxconverter_common
    import numpy as np

    if prototype is not None:
        if hasattr(prototype, 'shape') and hasattr(prototype, 'dtype'):
            shape = prototype.shape
            dtype = prototype.dtype.name
        else:
            raise TypeError("`prototype` has to be a valid `numpy.ndarray` of shape of your input")
    else:
        if not all([shape, dtype]):
            raise RuntimeError(
                "Did you forget to pass `prototype` or (`shape` & `dtype`)")
        try:
            dtype = np.dtype(dtype).name
        except TypeError:
            raise TypeError(
                '`dtype` not understood. '
                'It has to be a valid `np.dtype` or `np.dtype.type` object '
                'or an `str` that represents a valid numpy data type')
    if not isinstance(shape, tuple) or isinstance(shape, list):
        raise RuntimeError("Inferred `shape` attribute is not a tuple / list")
    return node_name, _get_tensortype(shape, dtype)


def is_installed(packages):
    if not isinstance(packages, list):
        packages = [packages]
    for p in packages:
        if importlib.util.find_spec(p) is None:
            return False
    return True

