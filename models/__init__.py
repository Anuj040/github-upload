import sys
import layers
import losses
import initializers


class Backbone(object):
    """
    Stores additional infromation on backbones
    """

    def __init__(self, backbone):
        # Dictionary mapping custom layer names to the correct classes

        self.custom_objects = {
            'UpsampleLike': layers.UpsampleLike,
            'PriorProbability': initializers.PriorProbability,
            'RegressBoxes': layers.RegressBoxes,
            'FilterDetections': layers.FilterDetections,
            'Anchors': layers.Anchors,
            'ClipBoxes': layers.ClipBoxes,
            '_focal': losses.focal(),
            'bce_': losses.bce(),
            'iou_':losses.iou(),
            }

        self.backbone = backbone
        self.validate()

    
    def validate(self):
        """
        Checks whether the backbone string is correct.
        """
        raise NotImplementedError('validate method not implemented.')

    def preprocess_image(self, inputs):
        """
        Takes as input an image and prepares it for being passed through the network.
        Having this function in Backbone allows other backbones to define a specific preprocessing step.
        """
        raise NotImplementedError('preprocess_image method not implemented.')