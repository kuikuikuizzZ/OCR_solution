import numpy as np
import cv2
import collections

REGION_TYPE = {
    'rect': 'N4',  # [N,4] array
    'quad': 'N8',  # [N,8] array
    'N4': 'N4',  # [N,4] array
    'N42': 'N42',  # [N,4,2] array
    'N-12': 'N-12',  # [N,-1,2] array
    'N8': 'N8'  # [N,8] array
}


def poly2bbox(poly):
    key_points = list()
    rotated = collections.deque(poly)
    rotated.rotate(1)
    for (x0, y0), (x1, y1) in zip(poly, rotated):
        for ratio in (1 / 3, 2 / 3):
            key_points.append(
                (x0 * ratio + x1 * (1 - ratio), y0 * ratio + y1 * (1 - ratio)))
    x, y = zip(*key_points)
    #     adjusted_bbox = (min(x), min(y), max(x) - min(x), max(y) - min(y))
    adjusted_bbox = (min(x), min(y), max(x), max(y))
    return key_points, adjusted_bbox


def quads2rects(quads, flags='3ratio'):
    '''
    transform quadrangles(x1,y1,x2,y2,x3,y3,x4,y4) to rect(x1,y1,x2,y2)
    '''
    if flags == 'default':
        boxes = np.stack([
            np.min(quads[:, ::2], axis=-1),
            np.min(quads[:, 1::2], axis=-1),
            np.max(quads[:, ::2], axis=-1),
            np.max(quads[:, 1::2], axis=-1)
        ],
                         axis=-1)
    elif flags == '3ratio':
        #         print('3ratio')
        boxes = []
        polys = quads.reshape([quads.shape[0], 4, 2])
        for poly in polys:
            key_point, box = poly2bbox(poly)
            boxes.append(box)
        boxes = np.array(boxes)
    return boxes


def rects2quads(boxes):
    quads = np.stack([
        boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 1], boxes[:, 2],
        boxes[:, 3], boxes[:, 0], boxes[:, 3]
    ],
                     axis=-1)
    return quads


def warpedPoints(points, M):
    #     cv2.perspectiveTransform(np.zeros([1,3,2]).astype(np.float),M)
    points = points.reshape((-1, 2)).astype(np.float64)
    warped_points = cv2.perspectiveTransform(np.expand_dims(points, axis=0), M)
    return warped_points


def boxesTypeTransform(inputs, inType, outType, **kwargs):
    '''
    inputs: inputs array
    '''
    if isinstance(inputs, np.ndarray):
        outputs = np.empty(inputs.shape)
    else:
        raise TypeError('inputs is not a numpy ndarray')
    if inType == "N4":
        outputs = N4Transform(inputs, outType, **kwargs)
    elif inType == "N8":
        outputs = N8Transform(inputs, outType, **kwargs)
    elif inType == "N42":
        outputs = N42Transform(inputs, outType, **kwargs)
    elif inType == "N-12":
        outputs = NM2Transform(inputs, outType, **kwargs)
    else:
        raise NotImplementedError('this input_type is not inplemented')
    return outputs


def N4Transform(inputs, outType, **kwargs):
    assert inputs.shape[1] == 4, 'inputs shape not match inType N4'
    if outType == 'N8':
        outputs = rects2quads(inputs)
    else:
        raise NotImplementedError('outType %s for inType N4 is \
                                  not inplemented' % (outType))
    return outputs


def N8Transform(inputs, outType, **kwargs):
    assert inputs.shape[1] == 8, 'inputs shape not match inType N8'
    if not True:
        pass
    else:
        raise NotImplementedError('outType %s for inType %s is \
                                  not inplemented' % (outType, inType))


def N42Transform(inputs, outType, **kwargs):
    assert inputs.shape[1:] == (4, 2), 'inputs shape not match inType N42'
    if not True:
        pass
    else:
        raise NotImplementedError('outType %s for inType %s is \
                                  not inplemented' % (outType, inType))


def NM2Transform(inputs, outType, **kwargs):
    assert inputs.shape[2] == 2, 'inputs shape not match inType N-12'
    if not True:
        pass
    else:
        raise NotImplementedError('outType %s for inType %s is \
                                  not inplemented' % (outType, inType))
