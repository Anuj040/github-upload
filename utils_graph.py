

import tensorflow as tf
import keras.backend as K

def resize_images(images, size, method = 'bilinear', align_corners = False):
    methods = {
        'bilinear': tf.image.ResizeMethod.BILINEAR,
        'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        'bicubic': tf.image.ResizeMethod.BICUBIC,
        'area': tf.image.ResizeMethod.AREA,
    }
    return tf.image.resize_images(images, size, methods[method], align_corners)

def shift(shape, stride, anchors):
    """
    Produce shifted anchors based on shape of the map and stride size.

    Args
        shape: Shape to shift the anchors over. (h,w)
        stride: Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.

    Returns
        shifted_anchors: (fh * fw * num_anchors, 4)
    """
    shift_x = (K.arange(0, shape[1], dtype=K.floatx()) + K.constant(0.5, dtype=K.floatx())) * stride
    shift_y = (K.arange(0, shape[0], dtype=K.floatx()) + K.constant(0.5, dtype=K.floatx())) * stride
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = K.reshape(shift_x, [-1])
    shift_y = K.reshape(shift_y, [-1])

    # (4, fh * fw)
    shifts = K.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)
    # (fh * fw, 4)
    shifts = K.transpose(shifts)
    number_anchors = K.shape(anchors)[0]

    # number of base points = fh * fw
    k = K.shape(shifts)[0]

    # (k=fh*fw, num_anchors, 4)
    shifted_anchors = K.reshape(anchors, [1, number_anchors, 4]) + K.cast(K.reshape(shifts, [k, 1, 4]), K.floatx())
    # (k * num_anchors, 4)
    shifted_anchors = K.reshape(shifted_anchors, [k * number_anchors, 4])

    return shifted_anchors
