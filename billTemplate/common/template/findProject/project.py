import numpy as np
import cv2
from yacs.config import CfgNode as CN

from ..utils.boxes import REGION_TYPE

defultConfig = CN()
defultConfig.type = 'center'
defultConfig.project = CN()
defultConfig.project.HOMOGRAPHY_THRESHOLD = 8


class Project(object):
    def __init__(self, region_type, pred_type):
        if region_type not in REGION_TYPE.keys():
            raise ValueError('type %s is not in REGION_TYPE' % region_type)
        if pred_type not in REGION_TYPE.keys():
            raise ValueError('type %s is not in REGION_TYPE' % pred_type)
        self.region_type = REGION_TYPE[region_type]
        self.pred_type = REGION_TYPE[pred_type]

    def __call__(self):
        raise NotImplementedError('matcher __call__ method is not implemented')


class FindProjects(object):
    def __init__(self, project=None, config=defultConfig):
        if project is None:
            self.project = self.build_project(config)
        else:
            self.project = project

    def build_project(self, config):
        # TODO: more generalize
        if config.type == 'center':
            from .centerProject import CenterProject
            return CenterProject(config=config.project)

    def __call__(self, refRegionDict, refCombinations, preds,
                 predCombinations):
        results, matrixes = list(), list()
        for refCombination, predCombination in zip(refCombinations,
                                                   predCombinations):
            result = self.project(refRegionDict, refCombination, preds,
                                  predCombination)
            matrixes.append(result['overlap'])
            results.append(result)
        index = _compareResults(matrixes)
        return results[index]


def _compareResults(results):
    results = np.array(results)
    index = np.argmax(results, axis=-1)
    return index


def warpedPoints(points, M):
    #     cv2.perspectiveTransform(np.zeros([1,3,2]).astype(np.float),M)
    points = points.reshape((-1, 2)).astype(np.float64)
    warped_points = cv2.perspectiveTransform(np.expand_dims(points, axis=0), M)
    return warped_points


def computeHomography(predCenter, refCenter, config):
    M, _ = cv2.findHomography(predCenter, refCenter, cv2.RANSAC,
                              config.HOMOGRAPHY_THRESHOLD)
    return M
