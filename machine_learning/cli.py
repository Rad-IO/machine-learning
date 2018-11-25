import json
import argparse
import logging
import sys
import time

from machine_learning.config import Config
from machine_learning.models import Model
from storage.drivers import PhotosStorageDriver
import storage.handler_ml as handler_ml


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
    parser.add_argument('storage-config', help='Path to the storage config '
                                               'file.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Whether to log the workings of the API to stderr')

    return parser.parse_args()


def initiate_models(config):
    models = []

    for name, data in config.models.items():
        models.append(Model.load_from_pb(data['pb_path'], data['labels'],
                                         name))

    return models


def run_forever(storage_handler, models):
    """

    Parameters
    ----------
    storage_handler: handler_ml.StorageHandler
    models: iterable of Model
    photos_driver: PhotosStorageDriver
    """
    while True:
        logging.info('Looking for a photo to classify...')
        requests = storage_handler.get_unused_requests()['requests']

        if len(requests) == 0:
            logging.info('None found. Going to sleep...')
            time.sleep(2)
            continue

        logging.info('Starting to process {} requests.'.format(len(requests)))

        for req in requests:
            pid = req['photo']
            results = {}

            photo_data = storage_handler.get_photo(pid)

            for model in models:
                probs = model.predict(photo_data)
                results[model.name] = probs

            storage_handler.upload_result(results, req['id'])


def main():
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    cnf = get_config(args.config)
    handler = handler_ml.StorageHandler(args.storage_config)

    run_forever(handler, initiate_models(cnf))


if __name__ == '__main__':
    main()
