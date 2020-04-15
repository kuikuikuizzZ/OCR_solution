import cv2
import numpy as np

from .adapter import RRect2FixHeightAdapter
from ..onnx_densenetCN import ONNXDensenetRecognizer
from ..onnx_craft import ONNXCraft

class ONNXCraftDensenetSpotter(object):
    def __init__(self,detect_path,recog_path,config=None):
        if config is None:
            self.recognizer_cfg = None
            self.detector_cfg =  None
            self.adapter_cfg = dict(boxes_shape=(4, 2), output_size=(300, 32))
        else:
            self.recognizer_cfg = config.recognizer
            self.detector_cfg = config.detector
            self.adapter_cfg = config.adapter
        self.config = config
        
        self.detector = ONNXCraft(detect_path,
                                  config=self.detector_cfg)
        self.recognizer = ONNXDensenetRecognizer(recog_path,
                                                 config=self.recognizer_cfg)
        self.adapter = RRect2FixHeightAdapter(**self.adapter_cfg)
        
    def __call__(self,image):
        image_temp = image.copy()
        boxes = self.detector.detect(image)
        adaptedBoxes = self.adapter(boxes,image_temp)
        texts = self.recognizer.predict_batch(adaptedBoxes)
        return texts,boxes
    
    def healthful(self):
        width = self.recognizer.config['width']
        height =  self.recognizer.config['height']
        test_sample = np.random.rand(1,height,width,3)
        try:
            self.recognizer.predict_batch(test_sample)
        except Exception as e:
            raise e
        return "ok model is healthy"