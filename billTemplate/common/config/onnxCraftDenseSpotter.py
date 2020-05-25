import os
from yacs.config import CfgNode as CN

detect_path = "/tmp/snapshot/detect/craft-b3hw.onnx"
recog_path = "/tmp/snapshot/recog/model300.onnx"

width = 600
path_cfg = dict(detect_path=detect_path, recog_path=recog_path)
_C = CN(path_cfg, new_allowed=True)
_C.model_cfg = CN()
_C.model_cfg.recognizer = CN(
    dict(height=32,
         width=width,
         batch_size=128,
         input_name=['the_input:0'],
         output_name=['y_pred/truediv:0'],
         device='cpu'))
_C.model_cfg.detector = CN(
    dict(input_name=['actual_input_1'],
         output_name=['output1'],
         canvas_size=768,
         mag_ratio=0.8,
         text_threshold=0.2,
         link_threshold=0.2,
         low_text=0.4,
         poly=True))
_C.model_cfg.adapter = CN(dict(boxes_shape=(4, 2), output_size=(width, 32)))

_C.return_coordinates = False
_C.in_shape_type = 'rect'
cfg = _C
