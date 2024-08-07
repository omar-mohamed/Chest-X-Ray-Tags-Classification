from __future__ import absolute_import, division

from visual_model_selector import ModelFactory
from generator import AugmentedImageSequence
from configs import argHandler  # Import the default arguments
from utils import set_gpu_usage, get_evaluation_metrics, custom_load_model
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
import os

FLAGS = argHandler()
FLAGS.setDefaults()

set_gpu_usage(FLAGS.gpu_percentage)

model_factory = ModelFactory()

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


train_generator = get_generator(FLAGS.train_csv)
test_generator = get_generator(FLAGS.test_csv)

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

def get_metrics_from_generator(generator,threshold=0.5, verbose=1):
    y_hat = visual_model.predict_generator(generator, steps=generator.steps, workers=FLAGS.generator_workers,
                                           max_queue_size=FLAGS.generator_queue_length, verbose=verbose)
    y = generator.get_y_true()
    get_evaluation_metrics(y_hat, y, FLAGS.classes, FLAGS.loss_function, threshold=threshold,image_names=generator.get_images_names(),save_path=os.path.join(FLAGS.save_model_path,'exact_match.csv'))

visual_model.compile(loss='binary_crossentropy',
                     metrics=[metrics.BinaryAccuracy(threshold=FLAGS.multilabel_threshold)])

# print("***************Train Metrics*********************")
# get_metrics_from_generator(train_generator, FLAGS.multilabel_threshold)
print("***************Test Metrics**********************")
get_metrics_from_generator(test_generator, FLAGS.multilabel_threshold)


