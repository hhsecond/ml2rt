======
MLutil
======


Machine learning utilities for model conversion, serialization, loading etc


* Free software: Apache Software License 2.0
* Documentation: https://mlutil.readthedocs.io.


Documentation
-------------

MLutil provides some convenient functions to convert, save & load machine learning models. It currently supports Tensorflow, PyTorch, Sklearn, Spark and ONNX but frameworks like xgboost, coreml are on the way.

Saving Tensorflow model
***********************

.. code-block:: python

    import tensorflow as tf
    from mlutil import save_tensorflow
    # train your model here
    sess = tf.Session()
    save_tensorflow(sess, path, output=['output'])

Saving PyTorch model
********************

.. code-block:: python

    # it has to be a torchscript graph made by tracing / scripting
    from mlutil import save_torch
    save_torch(torch_script_graph, path)

Saving ONNX model
*****************

.. code-block:: python

    from mlutil import save_onnx
    save_onnx(onnx_model, path)

Saving sklearn model
********************

.. code-block:: python

    from mlutil import save_sklearn
    prototype = np.array(some_shape, dtype=some_dtype)  # Equivalent to the input of the model
    save_sklearn(sklearn_model, path, prototype=prototype)

    # or

    # some_shape has to be a tuple and some_dtype has to be a np.dtype, np.dtype.type or str object
    save_sklearn(sklearn_model, path, shape=some_shape, dtype=some_dtype)

    # or

    # some_shape has to be a tuple and some_dtype has to be a np.dtype, np.dtype.type or str object
    inital_types = utils.guess_onnx_tensortype(shape=shape, dtype=dtype)
    save_sklearn(sklearn_model, path, initial_types=initial_types)

Saving sparkml model
********************

.. code-block:: python

    from mlutil import save_sparkml
    prototype = np.array(some_shape, dtype=some_dtype)  # Equivalent to the input of the model
    save_sparkml(spark_model, path, prototype=prototype)

    # or

    # some_shape has to be a tuple and some_dtype has to be a np.dtype, np.dtype.type or str object
    save_sparkml(spark_model, path, shape=some_shape, dtype=some_dtype)

    # or

    # some_shape has to be a tuple and some_dtype has to be a np.dtype, np.dtype.type or str object
    inital_types = utils.guess_onnx_tensortype(shape=shape, dtype=dtype)
    save_sparkml(spark_model, path, initial_types=initial_types)

Sklearn and sparkml models will be converted to ONNX first and then save to the disk. These models can be executed using ONNXRuntime, RedisAI etc. ONNX conversion needs to know the type of the input nodes and hence we have to pass shape & dtype or a prototype from where the utility can infer the shape & dtype or an initial_type object which is understood by the conversion utility. Frameworks like sparkml allows users to have heterogeneous inputs with more than one type. In such cases, use `guess_onnx_tensortypes` and create more than one initial_types which can be passed to save function as a list


Loading model & script
**********************

.. code-block:: python

    model = mlutil.load_model(path)

    script = mlutil.load_script(script)
