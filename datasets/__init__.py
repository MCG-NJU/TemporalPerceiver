"""
Dataset init.
"""

from .kineticsGEBD import build as build_kineticsGEBD

def build_dataset(split, args):
    if args.dataset == 'kineticsGEBD':
        return build_kineticsGEBD(split, args)
    raise ValueError(f'dataset {args.dataset} not implemented.')
