
import base64
import logging
import shlex
import subprocess
import sys
import tempfile
from collections import namedtuple
from io import BytesIO
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from PIL import Image

import re

logging.basicConfig(level=logging.ERROR)

import datetime
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State


import platform


from crop_api import ImageSaliencyModel, is_symmetric, parse_output, reservoir_sampling
from image_manipulation import join_images, process_image
    
def predict_image(image):
    # save image
    
    image.save(r'image_user.jpeg')


    BIN_MAPS = {"Darwin": "mac", "Linux": "linux"}

    HOME_DIR = Path("../").expanduser()

    sys.path.append(str(HOME_DIR / "src"))
    bin_dir = HOME_DIR / Path("./bin")
    bin_path = bin_dir / BIN_MAPS[platform.system()] / "candidate_crops"
    model_path = bin_dir / "fastgaze.vxm"
    data_dir = HOME_DIR / Path("./data/")
    data_dir.exists()

    model = ImageSaliencyModel(
        crop_binary_path=bin_path,
        crop_model_path=model_path,
    )

    plt.matplotlib.__version__

#     img_path = data_dir / "image_user.jpeg"
#     img_path.exists()

    img_path = Path('image_user.jpeg')

    output = model.plot_img_crops(img_path)
    
    output2, _ = process_image(img_path, model)
    
    
    return output, output2
     