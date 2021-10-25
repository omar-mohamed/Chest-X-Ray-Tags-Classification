import os
from tensorflow.keras.models import load_model
from utils import custom_save_model
from configs import argHandler  # Import the default arguments

FLAGS = argHandler()
FLAGS.setDefaults()

model = load_model(FLAGS.load_model_path)
model.summary()

custom_save_model(model, os.path.dirname(FLAGS.load_model_path), 'model_compressed')
