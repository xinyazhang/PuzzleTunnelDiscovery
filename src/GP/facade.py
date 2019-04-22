#!/usr/bin/env python3

import argparse
import pipeline

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')
    # In the order that users general use
    pipeline.init.setup_parser(subparsers)
    pipeline.add_puzzle.setup_parser(subparsers)
    pipeline.autorun.setup_parser(subparsers)
    # Occationally users want to run pipeline stages individually
    pipeline.preprocess_key.setup_parser(subparsers)
    pipeline.preprocess_surface.setup_parser(subparsers)
    pipeline.train.setup_parser(subparsers)
    pipeline.solve.setup_parser(subparsers)

    args = parser.parse_args()
    assert args.command in dir(pipeline)
    getattr(pipeline, args.command).run(args)

if __name__ == '__main__':
    main()
