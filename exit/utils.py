import os
import math
import copy
import torch
import shutil
import datetime
import numpy as np
import plotly.graph_objects as go
from utils.motion_process import recover_from_ric

def get_model(model):
    """
    Extraction of the base model from PyTorch's wrapped modules
    :param model:
    :return: model
    """
    if hasattr(model, 'module'):
        return model.module
    return model

# TODO: add standardization of the skeleton axis representation (for both datasets) for motion rendering via plotly;
#       add saving procedure for experiments with timestamps.