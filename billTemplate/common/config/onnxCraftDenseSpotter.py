import os
from yacs.config import CfgNode as CN

snapshot_dir = '/clever/code/WorkRelate/OCR_solution/snapshot'
path_cfg = dict(detect_path=os.path.join(snapshot_dir,'craft-b3hw.onnx'),
                   recog_path=os.path.join(snapshot_dir,'model300.onnx'))
_C = CN(path_cfg,new_allowed=True)
_C.model_cfg = CN()
_C.model_cfg.recognizer = CN(dict(height=32,
                               width=300,
                               batch_size=128,
                               input_name = ['the_input:0'],
                               output_name=['y_pred/truediv:0'],
                               device='cpu')
                  )
_C.model_cfg.detector = CN(dict( canvas_size=768,
                                mag_ratio = 0.8,
                                text_threshold = 0.7,
                                link_threshold = 0.4,
                                low_text = 0.4,
                                poly = True)
                )
_C.model_cfg.adapter = CN()

cfg=_C