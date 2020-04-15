import numpy as np
from collections import OrderedDict
from .utils.boxes import warpedPoints, quads2rects
from .utils.compute_overlap import compute_overlap


class Mapping(object):
    def __init__(self, target_type='N4', pred_type='N42'):
        self.target_type = target_type
        self.pred_type = pred_type

    def __call__():
        raise NotImplementedError('%s __call__ method is not implemented',
                                  type(self))


class TextBoxesMapping(Mapping):
    def __init__(self, target_type='N8', pred_type='N42', iou_thresh=0.2):
        self.target_type = target_type
        self.pred_type = pred_type
        self.iou_thresh = iou_thresh

    def __call__(self, targetDict, predResults, matrix):
        predBoxes, predTexts = predResults
        # TODO: replace for below
        if not isinstance(targetDict, OrderedDict):
            targetDict = OrderedDict(targetDict)
        targetBoxes = []
        keys = []
        for key, box in targetDict.items():
            targetBoxes.append(box)
            keys.append(key)
        targetBoxes = np.array(targetBoxes).astype(np.float64)
        if self.target_type == 'N8':
            targetRects = quads2rects(targetBoxes)
        else:
            targetRects = targetBoxes
        predBoxes = np.array(predBoxes)
        predTexts = np.array(predTexts)
        # warpping
        warpedBoxes = warpedPoints(predBoxes, matrix)
        warpedRects = quads2rects(warpedBoxes.reshape([-1, 8]))
        # overlap
        overlap = compute_overlap(targetRects, warpedRects)
        maxOverlap = np.max(overlap, axis=1)
        indexes = np.argmax(overlap, axis=1)
        boxes = predBoxes.take(indexes, axis=0)
        texts = predTexts.take(indexes, axis=0)
        self.filterIOU(maxOverlap, boxes, texts)
        result_dict = {
            key: (box, text)
            for key, box, text in zip(keys, boxes, texts)
        }
        return result_dict

    def filterIOU(self, overlaps, boxes, texts):
        for i, overlap in enumerate(overlaps):
            if overlap < self.iou_thresh:
                boxes[i] = np.zeros([4, 2])
                texts[i] = ''
