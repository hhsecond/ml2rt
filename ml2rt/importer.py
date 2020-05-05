import os
from pathlib import Path
from collections import namedtuple

from .exporter import save_tensorflow

model_container = namedtuple('model_container', field_names=['data', 'inputs', 'outputs'])


def load_model(path: str, tags=None, signature=None):
    """
    Return the binary data. If the input path is a directory of SavedModel from
    tensorflow, it converts the SavedModel to freezed model protbuf and then read
    it as binary. It also returns the input and output lists along with the binary
    model data in case of SavedModel.

    :param path: File path from where the native model or the rai models are saved
    :param tags: Tags for reading from SavedModel
    :param signature: SignatureDef for reading from SavedModel
    """
    path = Path(path)
    if path.is_dir():  # Expecting TF SavedModel
        import tensorflow as tf
        if tf.__version__ > '1.15.9':
            raise RuntimeError("Current tensorflow version must be 1.x (preferably 1.15)"
                               "even if the model is built with 2.x. If this that doesn't"
                               "work, follow the steps mentioned in this guide which uses tracing - "
                               "https://leimao.github.io/blog/Save-Load-Inference-From-TF2-Frozen-Graph/"
                               "\nBe warned that creating graph by using tracing might not give you"
                               "expected result if your graph is relying on dynamic ops internally")
        if tags is None:
            tags = ['serve']
        if signature is None:
            signature = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        graph = tf.Graph()
        sess = tf.Session(graph=graph)
        with graph.as_default():
            try:
                model = tf.saved_model.loader.load(sess=sess, tags=tags, export_dir=str(path))
            except Exception as e:
                raise RuntimeError("We could not load the provided SavedModel. It is probably"
                                   "caused by the tensorflow version difference (You might have"
                                   "used a different version of tensorflow for saving the model)."
                                   "Above stacktrace must have more information") from e
        # TODO: try with multiple input/output
        inputs = []
        for val in model.signature_def[signature].inputs.values():
            inputs.append(val.name.split(':')[0])
        outputs = []
        for val in model.signature_def[signature].outputs.values():
            outputs.append(val.name.split(':')[0])
        tmp_path = Path('model.pb')
        save_tensorflow(sess, str(tmp_path), outputs)
        with open(tmp_path, 'rb') as f:
            data = f.read()
        tmp_path.unlink()
        return model_container(data, inputs, outputs)
    else:
        with open(path, 'rb') as f:
            return f.read()


def load_script(path: str):
    """
    Load script is a convinient method that just reads the content from the file
    and returns it, as of now. But eventually can do validations using PyTorch's
    scirpt compile utility and clean up the input files for user etc
    """
    with open(path, 'rb') as f:
        return f.read()
