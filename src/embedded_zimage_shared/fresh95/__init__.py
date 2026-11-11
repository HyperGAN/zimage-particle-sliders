"""Explicit new cached fresh095 opt-in; original product routes stay unchanged."""
PROFILE='zimage-explicit-cached-fresh95-v1'
from .util import Checkpoint,file_pin

def save_prepared_cache(*args,**kwargs):
    from .cache import save_prepared_cache as save
    return save(*args,**kwargs)

def open_training(*args,**kwargs):
    from .api import Training
    return Training(*args,**kwargs)

def open_inference(*args,**kwargs):
    from .api import Inference
    return Inference(*args,**kwargs)
