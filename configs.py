class argHandler(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    _descriptions = {'help, --h, -h': 'show this super helpful message and exit'}

    def setDefaults(self):
        self.define('train_csv', './IU-XRay/training_set.csv',
                    'path to training csv containing the images names and the labels')
        self.define('test_csv', './IU-XRay/testing_set.csv',
                    'path to testing csv containing the images names and the labels')
        self.define('image_directory', './IU-XRay/images',
                    'path to folder containing the patient folders which containg the images')
        self.define('visual_model_name', 'MobileNetV2',
                    'select from (VGG16, VGG19, DenseNet121, DenseNet169, DenseNet201, Xception, ResNet50, InceptionV3, InceptionResNetV2, NASNetMobile, NASNetLarge, MobileNet, MobileNetV2)')
        self.define('use_chexnet_weights', False,
                    'use pre-trained chexnet weights. Note only works with DenseNet121. If you use this option without popping layers it will have the classifier intact')
        self.define('chexnet_weights_path', 'pretrained_models/chexnet_densenet121_weights.h5', 'chexnet weights path')
        self.define('image_target_size', (224, 224, 3), 'the target size to resize the image')
        self.define('num_epochs', 100, 'maximum number of epochs')
        self.define('csv_label_columns', ['Tags'], 'the name of the label columns in the csv')
        self.define('classes',
                    ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema',
                     'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                     'Fracture',
                     'Support Devices'],
                    'the names of the output classes')
        self.define('multi_label_classification', True,
                    'determines if this is a multi classification problem or not. It affects the loss function')
        self.define('classifier_layer_sizes', [],
                    'a list describing the hidden layers of the classifier. Example [10,0.4,5] will create a hidden layer with size 10 then dropout wth drop prob 0.4, then hidden layer with size 5. If empty it will connect to output nodes directly.')
        self.define('conv_layers_to_train', -1,
                    'the number of layers that should be trained in the visual model counting from the end. -1 means train all and 0 means freezing the visual model')
        self.define('use_imagenet_weights', True, 'initialize the visual model with pretrained weights on imagenet')
        self.define('pop_conv_layers', 0,
                    'number of layers to be popped from the visual model. Note that the imagenet classifier is removed by default so you should not take them into considaration')
        self.define('final_layer_pooling', 'avg', 'the pooling to be used as a final layer to the visual model')
        self.define('load_model_path', '',
                    'a path containing the checkpoints. If provided with load_model_name the system will continue the training from that point or use it in testing.')
        self.define('save_model_path', './saved_model',
                    'where to save the checkpoints. The path will be created if it does not exist. The system saves every epoch by default')
        self.define('save_best_model_only', True,
                    'Only save the best weights on validation loss')
        self.define('learning_rate', 1e-3, 'The optimizer learning rate')
        self.define('learning_rate_decay_factor', 0.1,
                    'Learning rate decay factor when validation loss stops decreasing')
        self.define('optimizer_type', 'Nadam', 'Choose from (Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam)')
        self.define('gpu_percentage', 0.95, 'gpu utilization. If 0 it will use the cpu')
        self.define('batch_size', 2, 'batch size for training and testing')
        self.define('multilabel_threshold', 0.5,
                    'The threshold from which to detect a class. Only used with multi label classification.')
        self.define('generator_workers', 8, 'The number of cpu workers generating batches.')
        self.define('generator_queue_length', 12, 'The maximum number of batches in the queue to be trained on.')
        self.define('minimum_learning_rate', 1e-8, 'The minimum possible learning rate when decaying')
        self.define('reduce_lr_patience', 2,
                    'The number of epochs to reduce the learning rate when validation loss is not decreasing')
        self.define('show_model_summary', True, 'A flag to show or hide the model summary')
        self.define('positive_weights_multiply', 1.0, 'Controls the class_weight ratio between 0 and 1. Higher value means higher weighting of positive samples. Only works if use_class_balancing is set to true')
        self.define('use_class_balancing', True, 'If set to true it will automatically balance the classes by settings class weights')

    def define(self, argName, default, description):
        self[argName] = default
        self._descriptions[argName] = description

    def help(self):
        print('Arguments:')
        spacing = max([len(i) for i in self._descriptions.keys()]) + 2
        for item in self._descriptions:
            currentSpacing = spacing - len(item)
            print('  --' + item + (' ' * currentSpacing) + self._descriptions[item])
        exit()
