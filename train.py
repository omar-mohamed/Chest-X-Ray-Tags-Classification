from __future__ import absolute_import, division

from visual_model_selector import ModelFactory
from generator import AugmentedImageSequence
from configs import argHandler  # Import the default arguments
from model_utils import get_optimizer, get_class_weights
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import os
from tensorflow.keras.models import load_model
from augmenter import augmenter

FLAGS = argHandler()
FLAGS.setDefaults()

model_factory = ModelFactory()


# load training and test set file names

def get_generator(csv_path, data_augmenter=None):
    return AugmentedImageSequence(
        dataset_csv_file=csv_path,
        label_columns=FLAGS.csv_label_columns,
        class_names=FLAGS.classes,
        multi_label_classification=FLAGS.multi_label_classification,
        source_image_dir=FLAGS.image_directory,
        batch_size=FLAGS.batch_size,
        target_size=FLAGS.image_target_size,
        augmenter=data_augmenter,
        shuffle_on_epoch_end=True,
    )


train_generator = get_generator(FLAGS.train_csv, augmenter)
class_weights = None
if FLAGS.use_class_balancing and FLAGS.multi_label_classification:
    class_weights = get_class_weights(train_generator.y, FLAGS.positive_weights_multiply)
test_generator = get_generator(FLAGS.test_csv)

# load classifier from saved weights or get a new one
if FLAGS.load_model_path != '' and FLAGS.load_model_path is not None:
    visual_model = load_model(FLAGS.load_model_path)
    if FLAGS.show_model_summary:
        visual_model.summary()
else:
    visual_model = model_factory.get_model(FLAGS)

opt = get_optimizer(FLAGS.optimizer_type, FLAGS.learning_rate)

if FLAGS.multi_label_classification:
    visual_model.compile(loss='binary_crossentropy', optimizer=opt,
                         metrics=[metrics.BinaryAccuracy(threshold=FLAGS.multilabel_threshold)])
    checkpoint = ModelCheckpoint(os.path.join(FLAGS.save_model_path, 'best_model.hdf5'), monitor='val_binary_accuracy',
                                 verbose=1,
                                 save_best_only=True, mode='max')
else:
    visual_model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    checkpoint = ModelCheckpoint(os.path.join(FLAGS.save_model_path, 'best_model.hdf5'), monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True, mode='max')
try:
    os.makedirs(FLAGS.save_model_path)
except:
    print("path already exists")

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=FLAGS.learning_rate_decay_factor, patience=FLAGS.reduce_lr_patience,
                      verbose=1, mode="min", min_lr=FLAGS.minimum_learning_rate),
    checkpoint,
    # TensorBoard(log_dir=os.path.join(FLAGS.save_model_path, "logs"), batch_size=FLAGS.batch_size)
]

visual_model.fit_generator(
    generator=train_generator,
    steps_per_epoch=train_generator.steps,
    epochs=FLAGS.num_epochs,
    validation_data=test_generator,
    validation_steps=test_generator.steps,
    workers=FLAGS.generator_workers,
    callbacks=callbacks,
    max_queue_size=FLAGS.generator_queue_length,
    class_weight=class_weights,
    shuffle=False
)
