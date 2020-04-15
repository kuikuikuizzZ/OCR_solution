# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Sergey Karayev
# --------------------------------------------------------

cimport cython
import numpy as np
cimport numpy as np


def computeDiagOverlap(
    np.ndarray[double, ndim=2] boxes,
    np.ndarray[double, ndim=2] query_boxes
):
    """
    Args
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float

    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
#     assert boxes.shape == query_boxes.shape 
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[double, ndim=1] overlaps = np.zeros((K), dtype=np.float64)
    cdef double iw, ih, querybox_area
    cdef double box_area
    cdef unsigned int k, n
    for k in range(K):
        querybox_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        box_area = ((boxes[k, 2] - boxes[k, 0] + 1) *
                 (boxes[k, 3] - boxes[k, 1] + 1))
        iw = (
            min(boxes[k, 2], query_boxes[k, 2]) -
            max(boxes[k, 0], query_boxes[k, 0]) + 1
        )
        if iw > 0:
            ih = (
                min(boxes[k, 3], query_boxes[k, 3]) -
                max(boxes[k, 1], query_boxes[k, 1]) + 1
            )
            if ih > 0:
                ua = np.float64( +
                   querybox_area+ box_area - iw * ih
                )
                overlaps[k] = iw * ih / ua
    return overlaps

def compute_overlap(
    np.ndarray[double, ndim=2] boxes,
    np.ndarray[double, ndim=2] query_boxes
):
    """
    Args
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float

    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[double, ndim=2] overlaps = np.zeros((N, K), dtype=np.float64)
    cdef double iw, ih, box_area
    cdef double ua
    cdef unsigned int k, n
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = np.float64(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def computeOverlapArea(
    np.ndarray[double, ndim=2] boxes,
    np.ndarray[double, ndim=2] query_boxes
):
    """
    Args
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float

    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[double, ndim=2] overlaps = np.zeros((N, K), dtype=np.float64)
    cdef double iw, ih, box_area
    cdef double ua
    cdef unsigned int k, n
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = np.float64(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih 
    return overlaps

def compute_inside_ratio(
    np.ndarray[double, ndim=2] boxes,
    np.ndarray[double, ndim=2] query_boxes
):
    """
    Args
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float

    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[double, ndim=2] overlaps = np.zeros((N, K), dtype=np.float64)
    cdef double iw, ih, box_area
    cdef double ua
    cdef unsigned int k, n
    for k in range(K):
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    area = np.float64(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) 
                    )
                    overlaps[n, k] = iw * ih / area
    return overlaps



def compute_1d_overlap(
    np.ndarray[double, ndim=2] boxes,
    np.ndarray[double, ndim=2] query_boxes,
    int dim=0):
    '''
    compute the overlap for boxes and query_boxes in one dimension
    Parameters
    ----------
    boxes: [N,4] numpy.array
    query_boxes: [M,4] numpy.array
    dim: int, compute dimension 
    ----------
    Return: 
    overlap: [N,M] a matrix of overlap for the boxes and query_boxes
    '''
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[double, ndim=2] overlaps = np.zeros((N, K), dtype=np.float64)
    cdef double intersect, union
    cdef double ua
    cdef unsigned int k, n
    for k in range(K):
        for n in range(N):
            if dim == 0:
                intersect = min(boxes[n, 2], query_boxes[k, 2]) -  \
                            max(boxes[n, 0], query_boxes[k, 0]) + 1
                if intersect > 0:
                    union =  max(boxes[n, 2], query_boxes[k, 2]) -  \
                             min(boxes[n, 0], query_boxes[k, 0]) + 1
                    if union > 0:
                        overlaps[n,k] = intersect/union
            if dim == 1 :
                intersect = min(boxes[n, 3], query_boxes[k, 3]) - \
                            max(boxes[n, 1], query_boxes[k, 1]) + 1
                if intersect > 0:
                    union = max(boxes[n, 3], query_boxes[k, 3]) - \
                            min(boxes[n, 1], query_boxes[k, 1]) + 1
                    if union > 0:
                        overlaps[n,k] = intersect/union            
    return overlaps
