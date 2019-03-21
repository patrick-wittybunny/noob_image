import cv2
import json
import numpy as np
# import cloudinary
import os
import api_practice.utils as utils

from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view

def adjustBrightness(image, percentage, scale):
    ycbImage = utils.loadImageYcb(image)
    beta = 75 * percentage / 100 * scale
    Ychannel, Cr, Cb = cv2.split(ycbImage)
    Ychannel = np.clip(Ychannel + beta, 0, 255)
    ycbImage = np.uint8(cv2.merge([Ychannel, Cr, Cb]))
    imbright = cv2.cvtColor(ycbImage, cv2.COLOR_YCrCb2BGR)
    # combined = np.hstack([image, imbright])
    return imbright


@api_view(['POST'])
def cooler(request):

    data = utils.parseRequest(request)
    url = data['image_url']
    # percentage = data['percentage']

    original = utils.url_to_image(url)
    image = np.copy(original)

    # Pivot points for X-Coordinates
    originalValue = np.array([0, 50, 100, 150, 200, 255])
    # Changed points on Y-axis for each channel
    # bCurve = np.array([0, 80 + 35, 150 + 35, 190 + 35, 220 + 35, 255]) 
    # rCurve = np.array([0, 20 - 20,  40 - 20,  75 - 20, 150 + 20, 255])
    bCurve = np.array([0, 80, 150, 190, 220, 255])
    rCurve = np.array([0, 20,  40,  75, 150, 255])

    # Create a LookUp Table
    fullRange = np.arange(0, 256)
    rLUT = np.interp(fullRange, originalValue, rCurve)
    bLUT = np.interp(fullRange, originalValue, bCurve)

    # print(original)
    # print(image)
    # print(image.shape)
    bChannel = image[:, :, 0]
    bChannel = cv2.LUT(bChannel, bLUT)
    image[:, :, 0] = bChannel

    # Get the red channel and apply the mapping
    rChannel = image[:, :, 2]
    rChannel = cv2.LUT(rChannel, rLUT)
    image[:, :, 2] = rChannel

    url = utils.image_to_url("results/cooler.jpg", image)
    # cv2.imwrite("results/cooler_%s.jpg" % ('result'), image)
    # return Response('', status=status.HTTP_200_OK)
    return Response({"image_url": url})


@api_view(['POST'])
def warmer(request):
    data = utils.parseRequest(request)
    url = data['image_url']
    # percentage = data['percentage']

    original = utils.url_to_image(url)
    image = np.copy(original)

   # Pivot points for X-Coordinates
    originalValue = np.array([0, 50, 100, 150, 200, 255])
    # Changed points on Y-axis for each channel
    # rCurve = np.array([0, 80 + 35, 150 + 35, 190 + 35, 220 + 35, 255]) 
    # bCurve = np.array([0, 20 ,  40 ,  75 , 150, 255])

    rCurve = np.array([0, 80, 150, 190, 220, 255])
    bCurve = np.array([0, 20,  40,  75, 150, 255])

    # Create a LookUp Table
    fullRange = np.arange(0, 256)
    rLUT = np.interp(fullRange, originalValue, rCurve)
    bLUT = np.interp(fullRange, originalValue, bCurve)

    bChannel = image[:, :, 0]
    bChannel = cv2.LUT(bChannel, bLUT)
    image[:, :, 0] = bChannel

    # Get the red channel and apply the mapping
    rChannel = image[:, :, 2]
    rChannel = cv2.LUT(rChannel, rLUT)
    image[:, :, 2] = rChannel

    url = utils.image_to_url("results/warmer.jpg", image)
    # cv2.imwrite("results/warmer_%s.jpg" % ('result'), image)
    # return Response('', status=status.HTTP_200_OK)
    return Response({"image_url": url})

@api_view(['POST'])
def brighten(request):
    data = utils.parseRequest(request)
    url = data['image_url']
    percentage = data['percentage']
    # name = url.split('.')[0]
    # image = cv2.imread(url)
    image = utils.url_to_image(url)

    imbright = adjustBrightness(image, percentage, 1)
    url = utils.image_to_url("results/bright_%s_%2.2f%%.jpg" % ('result', percentage), imbright)
    return Response({"image_url": url})

@api_view(['POST'])
def darken(request):
    data = utils.parseRequest(request)
    url = data['image_url']
    percentage = data['percentage']
    # name = url.split('.')[0]
    image = utils.url_to_image(url)
    imdark = adjustBrightness(image, percentage, -1)
    url = utils.image_to_url("results/dark%s_%2.2f%%.jpg" % ('result', percentage), imdark)
    return Response({"image_url": url})

@api_view(['POST'])
def oldify(request):
    data = utils.parseRequest(request)
    url = data['image_url']
    percentage = data['percentage']
    src_path = 'old_film_filter'
    src = cv2.imread('src/' + src_path + '.jpg')
    dst = utils.url_to_image(url)

    output = np.copy(dst)
    srcLab = np.float32(cv2.cvtColor(src, cv2.COLOR_BGR2LAB))

    dstLab = np.float32(cv2.cvtColor(dst, cv2.COLOR_BGR2LAB))
    outputLab = np.float32(cv2.cvtColor(output, cv2.COLOR_BGR2LAB))

    # Split the Lab images into their channels
    srcL, srcA, srcB = cv2.split(srcLab)
    dstL, dstA, dstB = cv2.split(dstLab)
    outL, outA, outB = cv2.split(outputLab)

    outL = dstL - dstL.mean()
    outA = dstA - dstA.mean()
    outB = dstB - dstB.mean()

    # scale the standard deviation of the destination image
    outL *= srcL.std() / dstL.std()
    outA *= srcA.std() / dstA.std()
    outB *= srcB.std() / dstB.std()

    # Add the mean of the source image to get the color
    outL = outL + srcL.mean()
    outA = outA + srcA.mean()
    outB = outB + srcB.mean()

    # Ensure that the image is in the range
    # as all operations have been done using float
    outL = np.clip(outL, 0, 255)
    outA = np.clip(outA, 0, 255)
    outB = np.clip(outB, 0, 255)

    # Get back the output image
    outputLab = cv2.merge([outL, outA, outB])
    outputLab = np.uint8(outputLab)

    output = cv2.cvtColor(outputLab, cv2.COLOR_LAB2BGR)

    output = output + 0.8 * output.std() * np.random.random(output.shape)

    # cv2.line(output, (450, 0), (450, 345), (0,0,0), thickness = 1, lineType=cv2.LINE_AA)

    cv2.imwrite('results/%s.jpg' % (src_path), output)
    return Response('', status=status.HTTP_200_OK)

@api_view(['POST'])
def ig_filter(request):
    data = utils.parseRequest(request)
    url = data['image_url']
    # percentage = data['percentage']

    original = utils.url_to_image(url)
    image = np.copy(original)

    # LOGKernel = np.array((
    #     [0.4038, 0.8021, 0.4038],
    #     [0.8021, -4.8233, 0.8021],
    #     [0.4038, 0.8021, 0.4038]), dtype="float")
    # LOG = cv2.filter2D(img, cv2.CV_32F, LOGKernel)
    # cv2.normalize(LOG, dst=LOG, alpha=0, beta=1,
    #         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

   # Pivot points for X-Coordinates
    originalValue = np.array([0, 28, 56, 85, 113, 141, 170, 198, 227, 255])
    # Changed points on Y-axis for each channel
    # rCurve = np.array([0, 80 + 35, 150 + 35, 190 + 35, 220 + 35, 255]) 
    # bCurve = np.array([0, 20 ,  40 ,  75 , 150, 255])

    bCurve = np.array([0, 16, 35, 64, 117, 163, 200, 222, 237, 249])
    rCurve = np.array([0, 24, 49, 98, 141, 174, 201, 223, 239, 255 ])
    gCurve = np.array([0, 38, 66, 104, 139, 175, 206, 226, 245, 255 ])

    # Create a LookUp Table
    fullRange = np.arange(0, 256)
    rLUT = np.interp(fullRange, originalValue, rCurve)
    bLUT = np.interp(fullRange, originalValue, bCurve)
    gLUT = np.interp(fullRange, originalValue, gCurve)

    # Get the blue channel and apply the mapping
    bChannel = image[:, :, 0]
    bChannel = cv2.LUT(bChannel, bLUT)
    image[:, :, 0] = bChannel

    # Get the green channel and apply the mapping
    gChannel = image[:, :, 1]
    gChannel = cv2.LUT(gChannel, gLUT)
    image[:, :, 2] = gChannel

    # Get the red channel and apply the mapping
    rChannel = image[:, :, 2]
    rChannel = cv2.LUT(rChannel, rLUT)
    image[:, :, 2] = rChannel

    sharpen = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="int")
    image = cv2.filter2D(image, -1, sharpen)

    url = utils.image_to_url("results/ig_filter.jpg", image)
    # cv2.imwrite("results/warmer_%s.jpg" % ('result'), image)
    # return Response('', status=status.HTTP_200_OK)
    return Response({"image_url": url})


