# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 11:57:09 2018

@author: Brad
"""
from polyssifier import Polyssifier
import argparse
import logging
from .logger import make_logger
from copy import deepcopy
import numpy as np
import os

logger = make_logger('polyssifier')

def make_argument_parser():
    '''
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('data', default='data.npy',
                        help='Data file name')
    parser.add_argument('label', default='labels.npy',
                        help='label file name')
    parser.add_argument('--level', default='info',
                        help='Logging level')
    parser.add_argument('--name', default='default',
                        help='Experiment name')
    parser.add_argument('--concurrency', default='1',
                        help='Number of allowed concurrent processes')

    return parser


def main(args):
    parser = make_argument_parser()
    args = parser.parse_args()

    if args.level == 'info':
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    data = np.load(args.data)
    label = np.load(args.label)
    labelcopy = deepcopy(label)

    logger.info(
        'Starting classification with {} workers'.format(args.concurrency))

    # If there are more than 50 unique labels, then it is most likely
    # a regression problem. Otherwise it is probably
    # a classification problem.
    do_regress = False
    if(len(np.unique(labelcopy)) > 50):
        do_regress = True
    poly = Polyssifier(data, label, do_regress=do_regress,
                       project_name=args.name,
                       concurrency=int(args.concurrency))
    poly.build()
    poly.run()
    report = poly.report
    report.plot_scores(os.path.join('polyr_' + args.name, args.name))
    report.plot_features(os.path.join('polyr_' + args.name, args.name))

if __name__ == '__main__':
    main()
