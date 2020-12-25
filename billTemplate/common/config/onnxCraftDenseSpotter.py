import os
import glob
from yacs.config import CfgNode as CN

recog_dir = os.environ.get("RECOG_DIR","")
detect_dir = os.environ.get("DETECT_DIR","")
if not recog_dir or not detect_dir:
    serving = os.environ.get("SERVING_NAME","serving")
    recog_model = os.environ.get("RECOG_MODEL_NAME","recog")
    detect_model = os.environ.get("DETECT_MODEL_NAME","detect")
    recog_version = os.environ.get("RECOG_VERSION","v1")
    detect_version = os.environ.get("DETECT_VERSION","v1")
    recog_dir = os.path.join("/mnt",serving,recog_model,recog_version)
    detect_dir = os.path.join("/mnt",serving,detect_model,detect_version)
detect_path = glob.glob(os.path.join(detect_dir,"*.onnx"))[0]
recog_path = glob.glob(os.path.join(recog_dir,"*.onnx"))[0]
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
print(cfg)
