# -*- coding: utf-8 -*-
#  borrowed from https://github.com/clovaai/CRAFT-pytorch/imgproc.py
import numpy as np
import cv2


def loadImage(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4: img = img[:, :, :3]
    img = np.array(img)

    return img


def normalizeMeanVariance(in_img,
                          mean=(0.485, 0.456, 0.406),
                          variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0],
                    dtype=np.float32)
    img /= np.array(
        [variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0],
        dtype=np.float32)
    return img


def denormalizeMeanVariance(in_img,
                            mean=(0.485, 0.456, 0.406),
                            variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w / 2), int(target_h / 2))
    return resized, ratio, size_heatmap


def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img

def equalizeHist(image):
    (b,g,r) = cv2.split(image) #通道分解
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    image = cv2.merge((bH,gH,rH),)#通道合成
    return image

def equalizeHistCLAHE(image):
    (b,g,r) = cv2.split(image) #通道分解
    tileGridSize = max(image.shape)//64
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(tileGridSize,tileGridSize))
    bH = clahe.apply(b)
    gH = clahe.apply(g)
    rH = clahe.apply(r)
    image = cv2.merge((bH,gH,rH),)
    return image