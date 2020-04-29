import os 
import sys
import difflib
import time

import cv2
import numpy as np
import onnxruntime as rt

from . import keysCN as keys
characters = keys.alphabet_union[1:]+'卐'

class ONNXDensenetRecognizer(object):
    def __init__(self,onnx_path,config=None,**kwargs):
        self.sess = rt.InferenceSession(onnx_path)
        if config is None:
            self.config = dict(height=32,
                               width=300,
                               batch_size=128,
                               input_name = ['the_input:0'],
                               output_name=['y_pred/truediv:0'],
                               device='cpu')
        else:
            self.config = config
    def predict_batch(self,img_list):
        return self._predict_batch(img_list)

    def _predict_batch(self,img_list):
        img_batch = self._batch_input(img_list)
        pred = self._predict(img_batch)
        out_list = []
        y_pred = pred[0]
        for i in range(len(img_list)):
            y_encode = np.array([y_pred[i,:,:]]) 
            out = _decode(y_encode,)  ##
            out_list.append(out)
        return out_list

    def _predict(self,img_batch):
        return self.sess.run(self.config['output_name'],
                {self.config['input_name'][0]:img_batch})

    def _batch_input(self,img_list):
        height = self.config['height']
        width = self.config['width']
        batch  = int(self.config['batch_size'])
        if self.config['device'] == 'gpu':
            img_batch = np.ones([batch,height,width,1],np.float32)*0.5
            length_imgs =  len(img_list)
            if length_imgs > batch:
                batch = batch << (length_imgs//batch)
                self.config['batch_size'] = batch
        elif self.config['device'] == 'cpu':
            batch  = len(img_list)
            if not batch:
                return []
        else:
            raise NotImplementedError ('device %s is not suppported'%config['device'])

        img_batch = np.ones([batch,height,width,1],np.float32)*0.5
        for i,img in enumerate(img_list):
            img_L = _normalize(img)
            img_batch[i,:,:width,:] = np.expand_dims(img_L,axis=-1)
        return img_batch

def _decode(pred):
    text = pred.argmax(axis=2)[0]
    length = len(text)
    char_list = []
    n = len(characters)-1
    for i in range(length):
        # 这里text[i] != n就是处理出来的分隔符，text[i - 1] == text[i]处理重复的字符
        if text[i] != n and (not (i > 0 and text[i - 1] == text[i])):
                char_list.append(characters[text[i]])
    return u''.join(char_list) 

def _normalize(img):
    img_L = img.astype(np.float32)
    img_L = img_L[:,:,0]*0.114 + img_L[:,:,1]* 0.587 + img_L[:,:,2]* 0.299 
    img_L = img_L/ 255.0 - 0.5
    return img_L