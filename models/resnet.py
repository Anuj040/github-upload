import os
import sys

import keras
import keras_resnet
import keras_resnet.models
import keras.layers as KL
from keras.utils import get_file

ROOT_DIR = os.path.abspath('../../')
sys.path.append(ROOT_DIR)

from FCOS.models import retinanet
from FCOS.models import Backbone

class ResNetBackbone(Backbone):
    """
    Describes backbone information and provides utility functions.
    """

    def __init(self, backbone):
        super(ResNetBackbone, self).__init__(backbone)
        self.custom_objects.update(keras_resnet.custom_objects)
    
    def retinanet(self, *args, **kwargs):
        """
        Returns a retinanet model using the relevant backbone.
        """  
        return resnet_retinanet(*args, backbone = self.backbone, **kwargs)

    def download_imagenet(self):
        """
        Downloads ImageNet weights and returns path to weights file.
        """
        resnet_filename = 'ResNet-{}-model.keras.h5'
        resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)
        depth = int(self.backbone.replace('resnet', ''))

        filename = resnet_filename.format(depth)
        resource = resnet_resource.format(depth)
        if depth == 50:
            checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
        elif depth == 101:
            checksum = '05dc86924389e5b401a9ea0348a3213c'
        elif depth == 152:
            checksum = '6ee11ef2b135592f8031058820bb9e71'
        else:
            raise ValueError('Unknown depth')

        return get_file(
            filename,
            resource,
            cache_subdir='models',
            md5_hash=checksum
        )

    def validate(self):
        """
        Checks whether the backbone string is correct.
        """
        allowed_backbones = ['resnet50', 'resnet101', 'resnet152']

        if self.backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone, allowed_backbones))

def resnet_retinanet(num_classes, weights, backbone = 'resent50', inputs = None, modifier = None, **kwargs):
    """
    Builds a retinanet model using resnet backbone.
    Args:
        num_classes: Number of prediction classes
        backbone: 'resnet50' or 'resnet101' or 'resnet152'
        inputs: Inputs to the network
        modifier: Function handler to modify the backbone before use
                 For eg.: can be used to freeze backbone layers
    Returns:
        RetinaNet model with a ResNet backbone.
    """
    #Default input
    if inputs is None:
        inputs = KL.Input(shape = (None, None, 3))

    #Creating the resnet backbone
    if backbone == 'resnet50':
        resnet = keras_resnet.models.ResNet50(inputs, include_top = False, freeze_bn = True)
    resnet.load_weights(weights, by_name=True, skip_mismatch=True)
    #Invoke modifier if given
    if modifier:
        resnet = modifier(resnet)
    
    #Creating the full model
    #resnet.outputs are [C2, C3, C4, C5]
  
    return retinanet.retinanet(inputs = inputs, num_classes = num_classes, backbone_layers = resnet.outputs[1:], **kwargs)

