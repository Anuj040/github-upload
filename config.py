


class Config(object):
    #Model Parameters
    MODEL = 'FCOS'
    #Image Parameters
    IMG_MIN = 800
    IMG_MAX = 1333
    RANDOM_TRANSFORM = False

    #Training Parameters
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 64
    FREEZE_BACKBONE = True
    IOU_THRESHOLD = 0.5
    SCORE_THRESHOLD = 0.05
    MAX_DETECTIONS = 20

    #Data Parameters
    DATASET_TYPE = "pascal"
    VOC_CLASSES = {
        'aeroplane': 0,
        'bicycle': 1,
        'bird': 2,
        'boat': 3,
        'bottle': 4,
        'bus': 5,
        'car': 6,
        'cat': 7,
        'chair': 8,
        'cow': 9,
        'diningtable': 10,
        'dog': 11,
        'horse': 12,
        'motorbike': 13,
        'person': 14,
        'pottedplant': 15,
        'sheep': 16,
        'sofa': 17,
        'train': 18,
        'tvmonitor': 19
        }

    #Annotation Paramters
    MIN_ANNOT = 15#Minimum (pixel)size of the annotation to keep

    #Model Parameters
    WEIGHTS = True#Default mode to load imagenet weights