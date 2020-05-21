import base64
import time
import os
import json

import cv2
import numpy as np
from flask_restful import Resource
from flask import request
from yacs.config import CfgNode as CN

from common.models.onnx_craft_densenet_spotter import ONNXCraftDensenetSpotter
from common.template.templater import TemplateImpl

default_cfg = CN(new_allowed=True)
dir_path = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(dir_path, 'common/config/onnxCraftDenseSpotter.py')
default_cfg.merge_from_file(config_path)


class InferenceImpl(object):
    def __init__(self, config=default_cfg):
        detect_path = config.detect_path
        recog_path = config.recog_path
        self.model = ONNXCraftDensenetSpotter(detect_path=detect_path,
                                              recog_path=recog_path,
                                              config=config.model_cfg)
        self.config = config

    def __call__(self, image):
        return self.model(image)

    def healthful(self):
        return self.model.healthful()


inferOps = InferenceImpl(config=default_cfg)


class InferenceHealth(Resource):
    def __init__(self):
        self.ops = inferOps

    def get(self):
        return self.ops.healthful()


class Inference(Resource):
    def __init__(self, config=default_cfg):
        self.config = config
        self.inferOps = inferOps

    def get(self):
        return {"inference available": "ok"}

    def post(self):
        try:
            result = self.inference()
        except Exception as e:
            return str(e) + "\n inference is not finished.", 500
        return result

    def inference(self):
        image_file = request.files["image_file"]
        solution_json = request.values.get('solution')
        refer_dict, target_dict = self.decodeTemplate(solution_json)
        image = np.asarray(bytearray(image_file.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        pred_texts, pred_boxes = self.inferOps(image)
        tpi = TemplateImpl(refer_dict=refer_dict, target_dict=target_dict)
        mapping_dict = tpi(pred_texts, pred_boxes)
        response_json = self.composeOutput(mapping_dict)
        return response_json

    def decodeTemplate(self, solution_json):
        solution_dict = json.loads(solution_json, encoding='utf-8')
        try:
            spec = solution_dict['spec']
            config = spec['config']
            customTemplateOCR = config['customTemplateOCR']
            references = customTemplateOCR['references']
            targets = customTemplateOCR['targets']
            refer_dict = self.collectCoord(references)
            target_dict = self.collectCoord(targets)
        except KeyError:
            raise KeyError('references or targets info is invalid')
        return refer_dict, target_dict

    def composeOutput(self, mapping_dict):
        target_regions = list()
        for key, item in mapping_dict.items():
            outArr, text = item
            if self.config.return_coordinates:
                coordinates = N42Arr2Coordinates(outArr)
                compose_item = dict(name=key,
                                    coordinates=coordinates,
                                    text=text)
            else:
                compose_item = dict(name=key, text=text)
            target_regions.append(compose_item)
        return dict(solution=dict(results=target_regions))

    def coordinates2box(self, coord_list):
        boxes = []
        for item in coord_list:
            boxes.extend([int(item['x']), int(item['y'])])
        if self.config['in_shape_type'] == 'rect':
            boxes = self.regular_rect(boxes)
        return boxes

    def regular_rect(self, boxes):
        x1 = min(boxes[::2])
        x2 = max(boxes[::2])
        y1 = min(boxes[1::2])
        y2 = max(boxes[1::2])
        return [x1, y1, x2, y1, x2, y2, x1, y2]

    def collectCoord(self, boxDictList):
        resultDict = dict()
        for boxDict in boxDictList:
            resultDict[boxDict['name']] = self.coordinates2box(
                boxDict['coordinates'])
        return resultDict


def N42Arr2Coordinates(outArray):
    coord_list = np.around(outArray).astype(np.int64).tolist()
    coordinates = list()
    for x, y in coord_list:
        coordinates.append(dict(x=x, y=y))
    return coordinates
