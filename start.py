import os
import sys
import argparse

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import keras


ROOT_DIR = os.path.abspath('../')
#Import project
sys.path.append(ROOT_DIR)
import losses
from FCOS.config import Config
import FCOS.model as modellib
DEFAULT_LOG_DIR = os.path.join(ROOT_DIR, 'logs')

class TrainConfig(Config):
    STEPS_PER_EPOCH = 1000



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fully Convoluted Single Shot Detector')
    parser.add_argument('command',
                        metavar="<command>",
                        help = "'train' or 'inference'")
    parser.add_argument('--backbone', help='Backbone used by retinanet.',
                        default='resnet50', type = str)
    parser.add_argument('--dataset', help = 'path to dataset',
                        metavar = 'path/to/VOC2012')
    parser.add_argument('--N', '-n', default = 10,
                        type = int, required = False,
                        metavar="No.of training epochs")
    parser.add_argument('--logs', required=False,
                        metavar='path/to/logs',
                        default = DEFAULT_LOG_DIR)
    parser.add_argument('--weights', '-w', default = None,
                        type = str, help = "'True' if reload weights")
    parser.add_argument('--reload', type = bool, help = 'If to resume training from stored epoch', default = False)
    parser.add_argument('--no-evaluation', help = 'Disable per epoch evaluation.', dest = 'evaluation', action = 'store_false')
    args = parser.parse_args()



    if args.command == 'train':
        config = TrainConfig()
        model = modellib.FCOS(config, mode = 'train', backbone = args.backbone, 
                                dataset = args.dataset, resume = args.reload, model_dir=args.logs)
        model.train(epochs = args.N, backbone_name = args.backbone, evaluation = args.evaluation)
    