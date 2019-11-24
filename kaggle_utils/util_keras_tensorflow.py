from tensorflow.python.client import device_lib
import os
import logging

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

def assert_cuda_paths():
    logging.info("Checking CUDA installation".format())
    assert "LD_LIBRARY_PATH" in os.environ, "Missing LD_LIBRARY_PATH environment variable"
    logging.info(os.environ["LD_LIBRARY_PATH"])
    assert "/usr/local/cuda-9.0/bin" in [p for p in os.environ['PATH'].split(':')]
    logging.info("confirmed '/usr/local/cuda-9.0/bin' in PATH")

