import os
import sys

import numpy as np
import keras
import keras.backend as K

ROOT_DIR = os.path.abspath('../../')
sys.path.append(ROOT_DIR)

# from FCOS.utils.compute_overlap import compute_overlap


class AnchorParameters:
    """
    The parameters that define how anchors are generated.

    Args
        sizes : List of sizes to use. Each size corresponds to one feature level.
        strides : List of strides to use. Each stride correspond to one feature level.
        ratios : List of ratios to use per location in a feature map.
        scales : List of scales to use per location in a feature map.
    """

    def __init__(self, sizes, strides, ratios, scales, interest_sizes):
        self.sizes = sizes
        self.strides = strides
        self.ratios = ratios
        self.scales = scales
        self.interest_sizes = interest_sizes

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


"""
The default anchor parameters.
"""
AnchorParameters.default = AnchorParameters(
    sizes=[32, 64, 128, 256, 512],
    strides=[8, 16, 32, 64, 128],
    ratios=np.array([0.5, 1, 2], K.floatx()),
    scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], K.floatx()),
    interest_sizes=[
        [-1, 64],
        [64, 128],
        [128, 256],
        [256, 512],
        [512, 1e8],
    ],
)

def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X scales w.r.t. a reference window.

    Args:
        base_size:
        ratios:
        scales:

    Returns:
        anchors: (num_anchors, 4)

    """
    if ratios is None:
        ratios = AnchorParameters.default.ratios

    if scales is None:
        scales = AnchorParameters.default.scales

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    # (num_anchors, )
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    # (num_anchors, )
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (cx, cy, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors

def guess_shapes(image_shape, pyramid_levels=(3, 4, 5, 6, 7)):
    """
    Guess shapes based on pyramid levels.

    Args
         image_shape: The shape of the image.
         pyramid_levels: A list of what pyramid levels are used.

    Returns
        A list of image shapes at each pyramid level.
    """
    image_shape = np.array(image_shape[:2])
    feature_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return feature_shapes

def compute_locations_per_level(h, w, stride):
    # [0, 8, 16]
    shifts_x = np.arange(0, w * stride, step=stride, dtype=np.float32)
    # [0, 8, 16, 24]
    shifts_y = np.arange(0, h * stride, step=stride, dtype=np.float32)
    shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
    # (h * w, )
    shift_x = shift_x.reshape(-1)
    # (h * w, )
    shift_y = shift_y.reshape(-1)
    locations = np.stack((shift_x, shift_y), axis=1) + stride // 2
    return locations

def compute_locations(feature_shapes, anchor_params=None):
    """

    Args:
        feature_shapes: list of (h, w)
        anchor_params: instance of AnchorParameters

    Returns:
        locations: list of np.array (shape is (fh * fw, 2))

    """
    if anchor_params is None:
        anchor_params = AnchorParameters.default
    fpn_strides = anchor_params.strides
    locations = []
    for level, (feature_shape, fpn_stride) in enumerate(zip(feature_shapes, fpn_strides)):
        h, w = feature_shape
        locations_per_level = compute_locations_per_level(
            h, w, fpn_stride
        )
        locations.append(locations_per_level)
    return locations

def compute_interest_sizes(num_locations_each_level, anchor_param=None):
    """

    Args:
        num_locations_each_level: list of int
        anchor_param:

    Returns:
        interest_sizes (np.array): (sum(fh * fw), 2)

    """
    if anchor_param is None:
        anchor_param = AnchorParameters.default
    interest_sizes = anchor_param.interest_sizes
    assert len(num_locations_each_level) == len(interest_sizes)
    tiled_interest_sizes = []
    for num_locations, interest_size in zip(num_locations_each_level, interest_sizes):
        interest_size = np.array(interest_size)
        interest_size = np.expand_dims(interest_size, axis=0)
        interest_size = np.tile(interest_size, (num_locations, 1))
        tiled_interest_sizes.append(interest_size)
    interest_sizes = np.concatenate(tiled_interest_sizes, axis=0)
    return interest_sizes

def anchors_for_shape(
        image_shape,
        pyramid_levels=None,
        anchor_params=None,
        shapes_callback=None,
	):
    """
    Generators anchors for a given shape.

    Args
        image_shape: The shape of the image.
        pyramid_levels: List of ints representing which pyramids to use (defaults to [3, 4, 5, 6, 7]).
        anchor_params: Struct containing anchor parameters. If None, default values are used.
        shapes_callback: Function to call for getting the shape of the image at different pyramid levels.

    Returns
        np.array of shape (N, 4) containing the (x1, y1, x2, y2) coordinates for the anchors.
    """

    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    if anchor_params is None:
        anchor_params = AnchorParameters.default

    if shapes_callback is None:
        shapes_callback = guess_shapes
    feature_map_shapes = shapes_callback(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(
            base_size=anchor_params.sizes[idx],
            ratios=anchor_params.ratios,
            scales=anchor_params.scales
        )
        shifted_anchors = shift(feature_map_shapes[idx], anchor_params.strides[idx], anchors)
        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors