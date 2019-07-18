
def load_model(path: str):
    """
    Return the binary data if saved with `as_native` otherwise return the dict
    that contains binary graph/model on `graph` key (Not implemented yet).
    :param path: File path from where the native model or the rai models are saved
    """
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
