import numpy as np
import cv2


class RRect2FixHeightAdapter(object):
    def __init__(self, boxes_shape=(4, 2), output_size=(300, 32)):
        self.boxes_shape = boxes_shape
        self.output_size = output_size

    def __call__(self, boxes, image):
        '''
        boxes: [N,4,2] rotated_boxes
        image: origin image
        return:
        rotated_image_segs: a list of img segs [N,H,W,3]
        '''
        width, height, ori_w, ori_h = bounding_length(
            boxes, fixed_height=self.output_size[1])
        target_points = struct_target_points(width, height)
        rotated_list = []
        for box, dst in zip(boxes, target_points):
            text_seg, src = crop_box(box, image)
            M, _ = cv2.findHomography(src, dst)
            rotated_seg = cv2.warpPerspective(text_seg,
                                              M,
                                              self.output_size,
                                              borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=tuple(
                                                  [255, 255, 255]))
            rotated_list.append(rotated_seg)

        return np.stack(rotated_list, axis=0)


def struct_target_points(width, height, blank=(10, 2)):
    x1 = np.ones(width.shape) * blank[0]
    y1 = np.ones(width.shape) * blank[1]
    x2 = x1 + width
    y2 = y1.copy()
    x3 = x2.copy()
    y3 = y2 + height
    x4 = x1
    y4 = y1 + height
    return np.stack([x1, y1, x2, y2, x3, y3, x4, y4],
                    axis=-1).reshape(-1, 4, 2)


def crop_box(box, image):
    x, y, w, h = cv2.boundingRect(box)
    coorded_box = np.zeros(box.shape)
    coorded_box[:, 0] = box[:, 0] - x
    coorded_box[:, 1] = box[:, 1] - y
    return image[y:y + h, x:x + w, :], coorded_box


def bounding_length(boxes, fixed_height=32):
    left = np.linalg.norm(boxes[:, 3, :] - boxes[:, 0, :], axis=-1)
    right = np.linalg.norm(boxes[:, 1, :] - boxes[:, 2, :], axis=-1)
    up = np.linalg.norm(boxes[:, 1, :] - boxes[:, 0, :], axis=-1)
    down = np.linalg.norm(boxes[:, 3, :] - boxes[:, 2, :], axis=-1)
    height = (left + right) / 2
    width = (up + down) / 2
    ratio = height / fixed_height
    width_rescale = width / ratio
    return width_rescale, fixed_height, width, height

