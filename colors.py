import cv2
import numpy as np

def rgb2ycrcb(im_rgb):
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    return im_ycrcb

def ycrcb2rgb(im_ycrcb):
    im_rgb = cv2.cvtColor(im_ycrcb, cv2.COLOR_YCR_CB2RGB)
    return im_rgb

def rgb2hls(im_rgb):
    im_hls = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2HLS_FULL)
    return im_hls

def hls2rgb(im_hls):
    im_rgb = cv2.cvtColor(im_hls, cv2.COLOR_HLS2RGB_FULL)
    return im_rgb