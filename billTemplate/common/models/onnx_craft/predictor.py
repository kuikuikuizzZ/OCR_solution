import cv2
import numpy as np
import onnxruntime as rt

from . import imgproc
from . import craft_utils


class ONNXCraft(object):
    def __init__(self, onnx_path, config=None):
        self.sess = rt.InferenceSession(onnx_path)

        if config is None:
            self.config = dict(canvas_size=768,
                               mag_ratio=0.8,
                               text_threshold=0.7,
                               link_threshold=0.4,
                               low_text=0.4,
                               poly=True)
        else:
            self.config = config

    def detect(self, image):
        image = imgproc.equalizeHist(image)
        img_resized, target_ratio, size_heatmap =  \
                imgproc.resize_aspect_ratio(image,
                                            self.config['canvas_size'],
                                            interpolation=cv2.INTER_LINEAR,
                                            mag_ratio=self.config['mag_ratio'])
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = x.transpose(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = np.expand_dims(x, axis=0)  # [c, h, w] to [b, c, h, w]
        # forward pass
        y = self._predict(x)
        y = y[0]

        # make score and link map
        score_text = y[0, :, :, 0]
        score_link = y[0, :, :, 1]

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link,
                                               self.config['text_threshold'],
                                               self.config['link_threshold'],
                                               self.config['low_text'],
                                               self.config['poly'])

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        return boxes

    def _predict(self, batch_input):
        return self.sess.run(self.config['output_name'],
                             {self.config['input_name'][0]: batch_input})
