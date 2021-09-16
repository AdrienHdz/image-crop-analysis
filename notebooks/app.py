import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import model




import base64
import logging
import shlex
import subprocess
import sys
import tempfile
from collections import namedtuple
from io import BytesIO
from pathlib import Path

#####
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
####


fig = plt.figure()

with open("notebooks/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Prediction of Saliency Maps'+' '+':world_map:')

st.markdown("Welcome to this simple web application that lets you predict saliency maps on any images. The model will highlight with red boxes the region on which people's eyes focus first. This is in Alpha stage, therefore some bugs might appear.")


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Predict")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        image = image.convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                time.sleep(1)
#                 st.success('Classified')
                fig1, fig2 = predict(image)
                st.write(fig1)
                st.write(fig2)


def predict(image):
#     classifier_model = "base_dir.h5"
#     IMAGE_SHAPE = (224, 224,3)
#     model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
#     test_image = image.resize((224,224))
#     test_image = preprocessing.image.img_to_array(test_image)
#     test_image = test_image / 255.0
#     test_image = np.expand_dims(test_image, axis=0)
#     class_names = [
#           'Backpack',
#           'Briefcase',
#           'Duffle', 
#           'Handbag', 
#           'Purse']
#     predictions = model.predict(test_image)
#     scores = tf.nn.softmax(predictions[0])
#     scores = scores.numpy()
#     results = {
#           'Backpack': 0,
#           'Briefcase': 0,
#           'Duffle': 0, 
#           'Handbag': 0, 
#           'Purse': 0
# }

    
#     result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 
    return model.predict_image(image)
    

if __name__ == "__main__":
    main()