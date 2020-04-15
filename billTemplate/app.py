import os
import sys
from flask import Flask
from flask_restful import Api

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billTemplate.apis.v1alpha1.api import registry_resource

app = Flask(__name__)

api = Api(app)

registry_resource(api)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9001)
