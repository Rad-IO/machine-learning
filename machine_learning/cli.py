import json
import argparse
import logging
import sys

from machine_learning.config import Config
from machine_learning.server import ConnectAPI


def get_config(path):
    logging.info('Reading config from {}.'.format(path))
    with open(path) as stream:
        raw = json.load(stream)

    c = Config()
    c.models = raw['models']
    c.paths = raw['paths']
    c.port = raw['port']

    return c


def parse_args():
    parser = argparse.ArgumentParser('Rad-I/O machine learning runner.')
    parser.add_argument('config', help='Path to the config file.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Whether to log the workings of the API to stderr')

    return parser.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    cnf = get_config(args.config)

    api = ConnectAPI(__name__, cnf)


if __name__ == '__main__':
    main()
