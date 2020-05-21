### visualize
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def draw_box(image, box, color, thickness=4):
    """ Draws a box on an image with a given color.

    Args:
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness,
                  cv2.LINE_AA)


def draw_boxes(image, boxes, color, thickness=2):
    """ Draws boxes on an image with a given color.

    Args:
        image     : The image to draw on.
        boxes     : A [N, 4] matrix (x1, y1, x2, y2).
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)


def draw_images(image_list, dpi=200, fig_size=(10, 10), n_col=None):
    if isinstance(image_list, list):
        if isinstance(image_list[0], np.ndarray):
            max_width = max([img.shape[1] for img in image_list])
        elif isinstance(image_list[0], Image.Image):
            max_width = max([img.size[0] for img in image_list])
        else:
            max_width = 1000
        n_col = max(1000 // max_width, 1) if n_col is None else n_col
        n_col = n_col if len(image_list) > n_col else len(image_list)
        n_row = len(image_list)//n_col+1 if len(image_list)%n_col != 0 else \
                len(image_list)//n_col
        min_size = min(fig_size)
        fig = plt.figure(figsize=(min_size * n_col, min_size * n_row))
        for i in range(1, len(image_list) + 1):
            img = image_list[i - 1]
            ax = fig.add_subplot(n_row, n_col, i)
            plt.imshow(img)
        plt.subplots_adjust(wspace=0, hspace=0)  # 修改子图之间的间隔
    elif isinstance(image_list, np.ndarray):
        img = Image.fromarray(image_list)
        fig = plt.figure(dpi=dpi)
        plt.imshow(img)
    elif isinstance(image_list, Image.Image):
        fig = plt.figure(dpi=dpi)
        img = image_list
        plt.imshow(img)
    else:
        raise TypeError('img list type invalid')
    return fig


def draw_caption(image, box, caption, fontScale=2, thickness=2):
    """ Draws a caption above the box in an image.

    Args:
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale, (0, 0, 0), thickness)


from PIL import Image, ImageDraw, ImageFont


def drawText(image, box, text, size=20):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    fillColor = "#ff0000"
    #     setFont = ImageFont.truetype('', 20)
    dirname = os.path.dirname(__file__)
    Font = ImageFont.truetype(os.path.join(dirname, 'HYDaSongJ.ttf'), size)
    draw.text((box[2], box[3] - size), text, font=Font, fill=fillColor)
    return np.array(image)
