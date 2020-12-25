import os
import sys
import argparse

from flask import Flask
from flask_restful import Api
from billTemplate.apis.v1alpha1.api import registry_resource
from billTemplate.resources import default_config

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-p',
                    '--port',
                    type=int,
                    default=9001,
                    help='serving port')

app = Flask(__name__)

api = Api(app)

registry_resource(api)

if __name__ == '__main__':
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)
