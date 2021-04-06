from __future__ import absolute_import, division

from visual_model_selector import ModelFactory
from configs import argHandler  # Import the default arguments
from utils import set_gpu_usage, custom_load_model
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
import os
import numpy as np
from gradcam import GradCAM
import cv2
from tqdm import tqdm
from classes import classes
from generator import AugmentedImageSequence


FLAGS = argHandler()
FLAGS.setDefaults()

def get_generator(csv_path, data_augmenter=None):
    return AugmentedImageSequence(
        dataset_csv_file=csv_path,
        label_columns=FLAGS.csv_label_columns,
        class_names=FLAGS.classes,
        source_image_dir=FLAGS.image_directory,
        batch_size=FLAGS.batch_size,
        target_size=FLAGS.image_target_size,
        augmenter=data_augmenter,
        shuffle_on_epoch_end=False,
    )

write_path = os.path.join(FLAGS.save_model_path,'cam_output')

try:
    os.makedirs(write_path)
except:
    print("path already exists")

set_gpu_usage(FLAGS.gpu_percentage)

model_factory = ModelFactory()

if FLAGS.load_model_path != '' and FLAGS.load_model_path is not None:
    base_name = os.path.basename(FLAGS.load_model_path)
    if '.' in base_name:
        visual_model = load_model(FLAGS.load_model_path)
    else:
        visual_model = custom_load_model(os.path.dirname(FLAGS.load_model_path),os.path.basename(FLAGS.load_model_path))
    if FLAGS.show_model_summary:
        visual_model.summary()
else:
    visual_model = model_factory.get_model(FLAGS)

FLAGS.batch_size = 1
test_generator = get_generator(FLAGS.test_csv)

images_names = test_generator.get_images_names()

top_k = 3

for batch_i in tqdm(range(test_generator.steps)):
    batch, _ = test_generator.__getitem__(batch_i)
    image_path = os.path.join(FLAGS.image_directory, images_names[batch_i])
    original = cv2.imread(image_path)
    preds = visual_model.predict(batch)

    predicted_classes = np.argpartition(preds[0], -top_k)[-top_k:]
    avg_heatmap = None
    avg_heatmap = np.zeros((original.shape[0],original.shape[1]),dtype=int)
    for predicted_class in predicted_classes:
        label = classes[predicted_class]
        cam = GradCAM(visual_model, predicted_class)
        heatmap = cam.compute_heatmap(batch)

        heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))

        avg_heatmap += heatmap

    avg_heatmap = np.array(avg_heatmap / top_k,dtype=np.uint8)

    (heatmap, output) = cam.overlay_heatmap(avg_heatmap, original, alpha=0.5)

    # cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
    # cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.8, (255, 255, 255), 2)

    cv2.imwrite(os.path.join(write_path,images_names[batch_i]),output)

