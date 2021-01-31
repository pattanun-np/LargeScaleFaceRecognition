import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2
import time

## Make canvas and set the color
def drawText(img, text, pos, fontSize, color=(255,255,255)):
    b,g,r = color
    fontpath = "./service/font/FC Iconic Bold.ttf"
    font = ImageFont.truetype(fontpath, fontSize)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos,  text, font = font, fill = (b, g, r))
    img = np.array(img_pil)
    return img

