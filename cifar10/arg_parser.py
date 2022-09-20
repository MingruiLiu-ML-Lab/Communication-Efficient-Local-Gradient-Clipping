"""
Command-line argument parsing.
"""

import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description='Distributed Training SGDClipGrad on CIFAR10 with Resnet26')
    parser.add_argument('--eta0', type=float, default=0.1,
                        help='Initial learning rate (default: 0.1).')
    parser.add_argument('--nesterov', action='store_true',
                        help='Use nesterov momentum (default: False).')
    parser.add_argument('--momentum', type=float, default=0,
                        help='Momentum used in optimizer (default: 0).')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='Weight decay used in optimizer (default: 0).')
    parser.add_argument('--step-decay-milestones', nargs='*', default=[],
                        help='Used for step decay denoting when to decrease the step size and the clipping parameter, unit in iteration (default: []).')   
    parser.add_argument('--step-decay-factor', type=float, default=0.1,
                        help='Step size and clipping paramter shall be multiplied by this on step decay (default: 0.1).')
    parser.add_argument('--clipping-param', type=float, default=1.0,
                        help='Weight decay used in optimizer (default: 1.0).')
    parser.add_argument('--clipping-option', type=str, default='max',
                        choices=['max', 'local', 'global_average'],
                        help='How to clip (default: max).')
    parser.add_argument('--baseline', action='store_true',
                        help='Do baseline local SGD (True) or not (False) (default: False).')  

    parser.add_argument('--world-size', type=int, default=8,
                        help='Number of processes in training (default: 8).')
    parser.add_argument('--rank', type=int, default=0,
                        help='Which process is this (default: 0).')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='Which GPU is used in this process (default: 0).')
    parser.add_argument('--init-method', type=str, default='file://',
                        help='URL specifying how to initialize the process group (default: file//).')  
    parser.add_argument('--communication-interval', type=int, default=8,
                        help='Number of train epochs (default: 8).')

    # Training
    parser.add_argument('--train-epochs', type=int, default=50,
                        help='Number of train epochs (default: 50).')
    parser.add_argument('--batchsize', type=int, default=32,
                        help='How many images in each train epoch for each GPU (default: 32).')
    parser.add_argument('--validation', action='store_true',
                        help='Do validation (True) or test (False) (default: False).')        
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Percentage of training samples used as validation (default: 0.1).')
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='How often should the model be evaluated during training, unit in epochs (default: 1).')

    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='Which dataset to run on (default: CIFAR10).')  
    parser.add_argument('--dataroot', type=str, default='../data',
                        help='Where to retrieve data (default: ../data).')
    parser.add_argument('--model', type=str, default='resnet56',
                        help='Which model to use (default: resnet56).')  
    parser.add_argument('--reproducible', action='store_true',
                        help='Ensure reproducibility (default: False).')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='Random seed (default: 0).')

    parser.add_argument('--log-folder', type=str, default='../logs',
                        help='Where to store results.')

    return parser.parse_args()
