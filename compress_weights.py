import os
from tensorflow.keras.models import model_from_json
import efficientnet.tfkeras
from tensorflow.keras.models import load_model
from utils import custom_save_model


model = load_model('./saved_model/chest_tags_efficientB4.hdf5')
model.summary()

custom_save_model(model,'./EfficientNetB4_100_16','chest_tags_effiecientB4')