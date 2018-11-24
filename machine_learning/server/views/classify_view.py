from flask.views import View
from flask import request, jsonify, Response

from machine_learning.server.utils import models

class ClassifyView(View):
    """Class representing the default entry point for the API.

    Path: /classify?id=<local_id_of_the_image_to_classify>.
    Methods: PUT

    Returns
    -------
    JSON
        Mapping condition names to the probability outputs of the neural
        network.

    """
    methods = ['PUT']
    expected_values = 'id'

    def dispatch_request(self):
        args = request.args.to_dict()
        img_id = args['id']

        #TODO: IMPLEMENT CONNECTION TO STORAGE, TO GET PHOTO

        for model in models:
            pass