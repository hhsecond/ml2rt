import time
import os
import sys

from ml2rt import load_model, load_script
from ml2rt import (
    save_tensorflow, save_torch, save_onnx, save_sklearn, save_sparkml)
from ml2rt import utils
import tensorflow as tf
import torch
from sklearn import linear_model, datasets
import onnx
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
import pyspark


def get_tf_graph():
    x = tf.placeholder(tf.float32, name='input')
    W = tf.Variable(5., name='W')
    b = tf.Variable(3., name='b')
    y = x * W + b
    y = tf.identity(y, name='output')


class MyModule(torch.jit.ScriptModule):
    def __init__(self):
        super(MyModule, self).__init__()

    @torch.jit.script_method
    def forward(self, a, b):
        return a + b


def get_sklearn_model_and_prototype():
    model = linear_model.LinearRegression()
    boston = datasets.load_boston()
    X, y = boston.data, boston.target
    model.fit(X, y)
    return model, X[0].reshape(1, -1).astype(np.float32)


def get_onnx_model():
    torch_model = torch.nn.ReLU()
    # maybe there exists, but couldn't find a way to pass
    # the onnx model without writing to disk
    torch.onnx.export(torch_model, torch.rand(1, 1), 'model.onnx')
    onnx_model = onnx.load('model.onnx')
    os.remove('model.onnx')
    return onnx_model


def get_spark_model_and_prototype():
    executable = sys.executable
    os.environ["SPARK_HOME"] = pyspark.__path__[0]
    os.environ["PYSPARK_PYTHON"] = executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = executable
    spark = SparkSession.builder.appName("redisai_test").getOrCreate()
    # label is input + 1
    data = spark.createDataFrame([
        (2.0, Vectors.dense(1.0)),
        (3.0, Vectors.dense(2.0)),
        (4.0, Vectors.dense(3.0)),
        (5.0, Vectors.dense(4.0)),
        (6.0, Vectors.dense(5.0)),
        (7.0, Vectors.dense(6.0))
    ], ["label", "features"])
    lr = LinearRegression(maxIter=5, regParam=0.0, solver="normal")
    model = lr.fit(data)
    prototype = np.array([[1.0]], dtype=np.float32)
    return model, prototype


class TestModel:
    # TODO: Detailed tests

    def test_TFGraph(self):
        get_tf_graph()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        path = f'{time.time()}.pb'
        save_tensorflow(sess, path, output=['output'])
        assert os.path.exists(path)
        os.remove(path)

    def test_PyTorchGraph(self):
        torch_graph = MyModule()
        path = f'{time.time()}.pt'
        save_torch(torch_graph, path)
        assert os.path.exists(path)
        os.remove(path)

    def test_ScriptLoad(self):
        dirname = os.path.dirname(__file__)
        path = f'{dirname}/testdata/script.txt'
        load_script(path)

    def test_ONNXGraph(self):
        onnx_model = get_onnx_model()
        path = f'{time.time()}.onnx'
        save_onnx(onnx_model, path)
        assert os.path.exists(path)
        load_model(path)
        os.remove(path)

    def test_SKLearnGraph(self):
        sklearn_model, prototype = get_sklearn_model_and_prototype()

        # saving with prototype
        path = f'{time.time()}.onnx'
        save_sklearn(sklearn_model, path, prototype=prototype)
        assert os.path.exists(path)
        load_model(path)
        os.remove(path)

        # saving with shape and dtype
        shape = prototype.shape
        if prototype.dtype == np.float32:
            dtype = prototype.dtype
        else:
            raise RuntimeError("Test is not configured to run with another type")
        path = f'{time.time()}.onnx'
        save_sklearn(sklearn_model, path, shape=shape, dtype=dtype)
        assert os.path.exists(path)
        load_model(path)
        os.remove(path)

        # saving with initial_types
        inital_types = utils.guess_onnx_tensortype(shape=shape, dtype=dtype)
        path = f'{time.time()}.onnx'
        save_sklearn(sklearn_model, path, initial_types=[inital_types])
        assert os.path.exists(path)
        load_model(path)
        os.remove(path)

    def test_SparkMLGraph(self):
        spark_model, prototype = get_spark_model_and_prototype()

        # saving with prototype
        path = f'{time.time()}.onnx'
        save_sparkml(spark_model, path, prototype=prototype)
        load_model(path)
        assert os.path.exists(path)
        os.remove(path)

        # saving with shape and dtype
        shape = prototype.shape
        if prototype.dtype == np.float32:
            dtype = prototype.dtype
        else:
            raise RuntimeError("Test is not configured to run with another type")
        path = f'{time.time()}.onnx'
        save_sparkml(spark_model, path, shape=shape, dtype=dtype)
        assert os.path.exists(path)
        load_model(path)
        os.remove(path)

        # saving with initial_types
        inital_types = utils.guess_onnx_tensortype(shape=shape, dtype=dtype)
        path = f'{time.time()}.onnx'
        save_sparkml(spark_model, path, initial_types=[inital_types])
        assert os.path.exists(path)
        load_model(path)
        os.remove(path)
