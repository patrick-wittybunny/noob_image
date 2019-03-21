import cv2
import numpy as np
import ssl
import cloudinary
import cloudinary.uploader
import cloudinary.api
import json
import os
from six.moves import urllib


def loadImageYcb(image):
    ycbImage = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycbImage = np.float32(ycbImage)
    return ycbImage

def parseRequest(request): 
    return json.loads(request.body.decode('utf-8'))

def url_to_image(url):
    resp = urllib.request.urlopen(url, context=ssl.SSLContext())
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return image

def image_to_url(path, image):
    cv2.imwrite(path, image)
    a = cloudinary.uploader.upload(path)
    os.remove(path)
    return a['url']
    # return ""
