import os
import sys
import datetime

import keras
import keras.models as KM
ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)

from FCOS.utils.transform import random_transform_generator
from FCOS.utils.image import random_visual_effect_generator
from FCOS.generators.voc_generator import VocGenerator
from FCOS.models.retinanet import retinanet_bbox
import losses
from callbacks import RedirectModel, Evaluate

def create_generators(config, dataset):
    """
    Create generators for training and validation.

    Args
        args: parseargs object containing configuration for generators.
        preprocess_image: Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size': config.BATCH_SIZE,
        # 'config': args.config,
        'image_min_side': config.IMG_MIN,
        'image_max_side': config.IMG_MAX
    }

    # create random transform generator for augmenting training data
    if config.RANDOM_TRANSFORM:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
            )
        visual_effect_generator = random_visual_effect_generator(
            contrast_range=(0.9, 1.1),
            brightness_range=(-.1, .1),
            hue_range=(-0.05, 0.05),
            saturation_range=(0.95, 1.05)
            )
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)
        visual_effect_generator = None

    train_generator = VocGenerator(
        config,
        dataset,
        'trainval',
        transform_generator=transform_generator,
        visual_effect_generator=visual_effect_generator,
        skip_difficult=True,
        **common_args
        )

    validation_generator = VocGenerator(
        config,
        dataset,
        'val',
        shuffle_groups=False,
        skip_difficult=True,
        **common_args
        )
    return train_generator, validation_generator

def model_with_weights(model, weights, skip_mismatch):
    """
    Load weights for model.

    Args
        model: The model to load weights for.
        weights: The weights to load.
        skip_mismatch: If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model

def freeze_model(model):
    """
    Set all layers in a model to non-trainable.

    The weights for these layers will not be updated during training.

    This function modifies the given model in-place,
    but it also returns the modified model to allow easy chaining with other functions.
    """
    for layer in model.layers:
        layer.trainable = False
    return model

def create_models(backbone_retinanet, num_classes, weights, num_gpus=0, freeze_backbone=False, lr=1e-5):
    """
    Creates three models (model, training_model, prediction_model).

    Args
        backbone_retinanet: A function to call to create a retinanet model with a given backbone.
        num_classes: The number of classes to train.
        weights: The weights to load into the model.
        num_gpus: The number of GPUs to use for training.
        freeze_backbone: If True, disables learning for the backbone.
        config: Config parameters, None indicates the default configuration.

    Returns
        model: The base model. This is also the model that is saved in snapshots.
        training_model: The training model. If num_gpus=0, this is identical to model.
        prediction_model: The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """

    modifier = freeze_model if freeze_backbone else None

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    model = backbone_retinanet(num_classes, weights = weights, modifier=modifier)
    
    training_model = model

    # make prediction model
    prediction_model = retinanet_bbox(model=model)

    # compile model
    training_model.compile(
        loss={
            'regression': losses.iou(),
            'classification': losses.focal(),
            'centerness': losses.bce(),
        },
        optimizer=keras.optimizers.adam(lr=lr)
    )

    return model, training_model, prediction_model

def create_callbacks(config, backbone, model, training_model, prediction_model, validation_generator, evalu, tensorboard_dir):
    """ Creates the callbacks to use during training.

    Args
        model: The base model.
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None

    if tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=0,
            batch_size=config.BATCH_SIZE,
            write_graph=True,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None
        )
        callbacks.append(tensorboard_callback)

    if (evalu) and (validation_generator is not None):
        evaluation = Evaluate(config, validation_generator, tensorboard=tensorboard_callback)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    # save the model
    if tensorboard_dir:
        # ensure directory created first; otherwise h5py will error after epoch.
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                tensorboard_dir,
                '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone=backbone,
                                                                    dataset_type=config.DATASET_TYPE)
            ),
            verbose=1,
            # save_best_only=True,
            # monitor="mAP",
            # mode='max'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=2,
        verbose=1,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=0
    ))

    return callbacks

class FCOS:


    def __init__(self, config, mode, backbone, dataset, model_dir, resume = False):

        self.config = config
        self.mode = mode
        self.dataset = dataset
        self.model_dir =  model_dir
        self.set_log_dir()
        self.build(config, backbone, resume)

    def build(self, config, backbone_name, resume):
        if 'resnet' in backbone_name:
            from FCOS.models.resnet import ResNetBackbone as b
            self.backbone = b(backbone_name)
    
        if resume:
            print("Loading model, may take some time.")
            self.model = self.load_model(self.model_dir, self.backbone)
            self.prediction_model = retinanet_bbox(model = self.model)
        else:        
            if self.config.WEIGHTS:
                weights = self.backbone.download_imagenet()
            print('Creating model, this may take a second...')
            self.model, self.training_model, self.prediction_model = create_models(
                    backbone_retinanet=self.backbone.retinanet,
                    num_classes=len(config.VOC_CLASSES),
                    weights=weights,
                    freeze_backbone=config.FREEZE_BACKBONE,
                    lr=self.config.LEARNING_RATE
                    )

    def train(self, epochs, backbone_name, evaluation):
        #Compile model
        self.model.compile(
            loss={
                'regression': losses.iou(),
                'classification': losses.focal(),
                'centerness': losses.bce(),
            },
            optimizer=keras.optimizers.adam(lr=1e-5)
            # optimizer=keras.optimizers.sgd(lr=1e-5, momentum=0.9, decay=1e-5, nesterov=True)

            )
        # create the generators
        train_generator, validation_generator = create_generators(self.config, self.dataset)
                    
        # create the callbacks
        callbacks = create_callbacks(
            self.config,
            backbone_name,
            self.model,
            self.training_model,
            self.prediction_model,
            validation_generator,
            evaluation,
            self.log_dir,
            )
            # start training
        return self.training_model.fit_generator(
            generator=train_generator,
            initial_epoch=0,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks,
            max_queue_size=10,
            validation_data=validation_generator
        )

    def load_model(self, filepath, backbone_name):
        """ Loads a retinanet model using the correct custom objects.

        Args
            filepath: one of the following:
                - string, path to the saved model, or
                - h5py.File object from which to load the model
            backbone_name         : Backbone with which the model was trained.

        Returns
            A keras.models.Model object.

        Raises
            ImportError: if h5py is not available.
            ValueError: In case of an invalid savefile.
        """
        return KM.load_model(filepath, custom_objects=self.backbone.custom_objects)


    def set_log_dir(self, model_path = None):
        """Sets the model directory and the epoch counter.
        model_path: If None, or a format mismatch, then set a new directory
        and start epochs from 0. Otherwise, extract the log directory and the 
        epoch counter from the file name.       
        
        """
        import re

        #Assume a start from beginning
        self.epoch = 0
        now = datetime.datetime.now()

        #If model path with date and epoch exists, use it.
        if model_path:
            #Get epoch and date from the file name
            #Sample path for windows
            # \path\to\logs_pix2pix\facades_20200607T1304\model_1_0001.h5

            split = re.split(r'\\', model_path)
            now = [a for a in split if a.startswith(self.config.MODEL + '_')][0]#Existing model's date and time
            
            regex = r".*[\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})"
            m = re.match(regex,now)

            e = [a for a in split if a.startswith('model_'+ str(self.config.CLASS_OBJECT) + '_')][0]#Last epoch
            regex = r".*[\w-]+(\d{4})\.h5"
            e = re.match(regex, e)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                                
                #Epoch No. in file 1-based, in Keras 0-based.
                #Adjust by one to start from the next epoch
                self.epoch = int(e.group(1))
                print('Resuming from epoch %d' % (self.epoch + 1))
        #Directory for train logs
        self.log_dir = os.path.join(self.model_dir, "{}_{:%Y%m%dT%H%M}".format(
                                    self.config.MODEL, now))
