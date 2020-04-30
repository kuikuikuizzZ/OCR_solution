import numpy as np
from copy import deepcopy

from .project import Project, warpedPoints, computeHomography
from ..utils.boxes import rects2quads, quads2rects
from ..utils.compute_overlap import computeDiagOverlap


class CenterProject(Project):
    def __init__(self, region_type='N8', pred_type='N42', config=None):
        super(CenterProject, self).__init__(region_type=region_type,
                                            pred_type=pred_type)
        self.config = config
        if self.config is None:
            from yacs.config import CfgNode as CN
            self.config = CN()
            self.config.HOMOGRAPHY_THRESHOLD = 6

    def __call__(self, refRegionDict, refCombination, preds, predCombination):
        refCenters, refBoxes = self.combineRefCenters(refRegionDict,
                                                      refCombination)
        predCenters, predBoxes = self.combinePredictCenters(
            preds, predCombination)
        M = computeHomography(predCenters, refCenters, self.config)
        warpedBoxes = warpedPoints(predBoxes, M)
        warpedBoxes = warpedBoxes.reshape([-1, 8])
        #     vertifyClockwise(warpedPredPoints)
        warpedRects = quads2rects(warpedBoxes, 'default')
        refRects = quads2rects(refBoxes, 'default')
        overlap = computeDiagOverlap(refRects, warpedRects).sum()

        return dict(overlap=overlap,
                    warpedRects=warpedRects,
                    refRects=refRects,
                    refBoxes=refBoxes,
                    predBoxes=predBoxes,
                    matrix=M)

    def combinePredictCenters(self, preds, combine):
        '''
        combineRects:   list of np.array(N,4,2)
        combine : a list of index
        out:
            combineCenters: list of np.array(N,2)
            combineBoxes:   list of np.array(N,4,2)
        '''

        predCenters = self.center(preds, flags='pred')
        combineCenters = np.take(predCenters, combine, axis=0)
        combineBoxes = np.take(preds, combine, axis=0)
        # TODO: unify the shape
        #         combineBoxes = toQuads(combineBoxes,self.pred_type)
        return combineCenters, combineBoxes

    def combineRefCenters(self, refRegionDict, refCombination):
        '''
        in : refRegionDict: {key:[x1,y1,x2,y2]}
             reCombination: a list of key
        out:
            combineCenters: list of np.array(N,2)
            combineRects:   list of np.array(N,4)
        '''
        refs = []
        for key in refCombination:
            refs.append(np.array(refRegionDict[key]))


#         refBoxes = rects2quads(np.array(refRects)).astype(np.float64)
        combineBoxes = np.stack(refs, axis=0).astype(np.float64)
        combineCenters = self.center(combineBoxes, flags='refer')
        # TODO: unify the shape
        if self.region_type == "N4":
            combineBoxes = rects2quads(combineBoxes)
        return combineCenters, combineBoxes

    def center(self, boxes, flags):
        if flags == 'pred':
            center_type = self.pred_type
        else:
            center_type = self.region_type
        if center_type == 'N42':
            return centerPoints(boxes)
        if center_type == 'N8' or center_type == 'N4':
            return centerBoxes(boxes)


def centerRect(point):
    '''
    Parameters
    ----------
    point: a [4] array 
    ----------
    Return: 
    center: [2] array
    '''
    x_center = np.mean(point[::2])
    y_center = np.mean(point[1::2])
    center = np.array([x_center, y_center])
    return center


def centerBoxes(points):
    '''
    Parameters
    ----------
    point: a [N,4(2)(8)] array 
    ----------
    Return: 
    center: [N,2] array
    '''
    x_center = np.mean(points[:, ::2], axis=-1)
    y_center = np.mean(points[:, 1::2], axis=-1)
    centers = np.stack([x_center, y_center], axis=-1)
    return centers


def centerPoints(boxes):
    '''
    Parameters
    ----------
    point: a [N,4,2] array 
    ----------
    Return: 
    center: [N,2] array
    '''
    x_centers = np.mean(boxes[:, :, 0], axis=1)
    y_centers = np.mean(boxes[:, :, 1], axis=1)
    return np.stack([x_centers, y_centers], axis=-1)
