from flask import Flask

import logging

import machine_learning.server.utils as utils
from machine_learning.models import Model


class ConnectAPI(Flask):
    """REST-based API connecting the ML module to the outer world.

    Parameters
    ----------
    include_name: str
        The name of the environment where the API is running.
    config: machine_learning.config.Config
        With the config of the API.

    """

    def __init__(self, include_name, config, *args, **kwargs):
        super(ConnectAPI, self).__init__(self, include_name, *args, **kwargs)
        utils.config = config
        self._setup_models(config)

    def _setup_models(self, config):
        utils.models =[
            Model.load_from_pb(m['pb_path'], m['labels'])
            for m in config.models
        ]

    def _setup_views(self):

        # TODO: ADD HERE
        pass
