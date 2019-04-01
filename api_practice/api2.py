import cv2
import json
import numpy as np
import dlib
import api_practice.utils as utils
import api_practice.image_utils as image_utils
import api_practice.video_utils as video_utils
import random
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view
import api_practice.faceBlendCommon as fbc


PREDICTOR_PATH = 'common/shape_predictor_68_face_landmarks.dat'
faceDetector = dlib.get_frontal_face_detector()
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

@api_view(['POST'])
def face_average(request):
  data = utils.parseRequest(request)
  # image_urls = data['image_urls']
  image_urls = [data['src_img'], data['dst_img']]
  # image_urls.append(data['src_img'])
  # image_urls.append(data['src_img'])

  images = []
  allPoints = []

  for url in image_urls:
    try:
      im = utils.url_to_image(url)
      if im is None:
        print("Unable to load image url")
        next
      else:
        utils.image_to_url("results/face_to_average/%d.jpg" % random.randint(0, 10000), im)
        points = fbc.getLandmarks(faceDetector,landmarkDetector, im)
        if(len(points) > 0):
          allPoints.append(points)
          im = np.float32(im)/255.0
          images.append(im)
        else:
          print("No face detected")
    except:
      print("Forbidden image")
  if len(images) == 0:
    print("No images loaded")
    return Response("no faces to average")

  w = 300
  h = 300

  boundaryPts = fbc.getEightBoundaryPoints(w, h)

  numImages = len(images)
  numLandmarks = len(allPoints[0])

  imagesNorm = []
  pointsNorm = []

  pointsAvg = np.zeros((numLandmarks, 2), dtype=np.float32)

  # Warp images and trasnform landmarks to output coordinate system,
  # and find average of transformed landmarks.
  for i, img in enumerate(images):

    points = allPoints[i]
    points = np.array(points)

    img, points = fbc.normalizeImagesAndLandmarks((h, w), img, points)

    # Calculate average landmark locations
    pointsAvg = pointsAvg + (points / (1.0*numImages))

    # Append boundary points. Will be used in Delaunay Triangulation
    points = np.concatenate((points, boundaryPts), axis=0)

    pointsNorm.append(points)
    imagesNorm.append(img)

  # Append boundary points to average points.
  pointsAvg = np.concatenate((pointsAvg, boundaryPts), axis=0)

  # Delaunay triangulation
  rect = (0, 0, w, h)
  dt = fbc.calculateDelaunayTriangles(rect, pointsAvg)

  # Output image
  output = np.zeros((h, w, 3), dtype=np.float)

  # Warp input images to average image landmarks
  for i in range(0, numImages):

    imWarp = fbc.warpImage(
      imagesNorm[i], pointsNorm[i], pointsAvg.tolist(), dt)

    # Add image intensities for averaging
    output = output + imWarp

  # Divide by numImages to get average
  output = output / (1.0*numImages)
  output = output * 255.0
  output = np.uint8(output)
  print(output)
  url = utils.image_to_url("results/face_to_average/face_average.jpg", output)
  return Response({"image_url": url})


def face_average2(request):
  data = utils.parseRequest(request)
  image_urls = data['image_urls']
  # image_urls = [data['src_img'], data['dst_img']]

  images = []
  allPoints = []

  for url in image_urls:
    try:
      im = utils.url_to_image(url)
      if im is None:
        print("Unable to load image url")
        next
      else:
        utils.image_to_url("results/face_to_average/%d.jpg" %
                           random.randint(0, 10000), im)
        points = fbc.getLandmarks(faceDetector, landmarkDetector, im)
        if(len(points) > 0):
          allPoints.append(points)
          im = np.float32(im)/255.0
          images.append(im)
        else:
          print("No face detected")
    except:
      print("Forbidden image")
  if len(images) == 0:
    print("No images loaded")
    return Response("no faces to average")

  w = 300
  h = 300

  boundaryPts = fbc.getEightBoundaryPoints(w, h)

  numImages = len(images)
  numLandmarks = len(allPoints[0])

  imagesNorm = []
  pointsNorm = []

  pointsAvg = np.zeros((numLandmarks, 2), dtype=np.float32)

  # Warp images and trasnform landmarks to output coordinate system,
  # and find average of transformed landmarks.
  for i, img in enumerate(images):

    points = allPoints[i]
    points = np.array(points)

    img, points = fbc.normalizeImagesAndLandmarks((h, w), img, points)

    # Calculate average landmark locations
    pointsAvg = pointsAvg + (points / (1.0*numImages))

    # Append boundary points. Will be used in Delaunay Triangulation
    points = np.concatenate((points, boundaryPts), axis=0)

    pointsNorm.append(points)
    imagesNorm.append(img)

  # Append boundary points to average points.
  pointsAvg = np.concatenate((pointsAvg, boundaryPts), axis=0)

  # Delaunay triangulation
  rect = (0, 0, w, h)
  dt = fbc.calculateDelaunayTriangles(rect, pointsAvg)

  # Output image
  output = np.zeros((h, w, 3), dtype=np.float)

  # Warp input images to average image landmarks
  for i in range(0, numImages):

    imWarp = fbc.warpImage(
        imagesNorm[i], pointsNorm[i], pointsAvg.tolist(), dt)

    # Add image intensities for averaging
    output = output + imWarp

  # Divide by numImages to get average
  output = output / (1.0*numImages)
  output = output * 255.0
  output = np.uint8(output)
  print(output)
  url = utils.image_to_url("results/face_to_average/face_average.jpg", output)
  return Response({"image_url": url})

@api_view(['POST'])
def face_morph(request):
  data = utils.parseRequest(request)
  dst_url = data['dst_url']
  src_url = data['src_url']

  src_img = utils.url_to_image(src_url)
  dst_img = utils.url_to_image(dst_url)

  utils.image_to_url("results/face_morph/morph_orig_src.jpg", src_img)
  utils.image_to_url("results/face_morph/morph_orig_dst.jpg", dst_img)

  src_pts = fbc.getLandmarks(
      faceDetector, landmarkDetector, cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))
  dst_pts = fbc.getLandmarks(
      faceDetector, landmarkDetector, cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB))

  src_pts = np.array(src_pts)
  dst_pts = np.array(dst_pts)

  # Convert image to floating point in the range 0 to 1
  src_img = np.float32(src_img)/255.0
  dst_img = np.float32(dst_img)/255.0


  h = 300
  w = 300

  # Normalize image to output coordinates.
  srcNorm, src_pts = fbc.normalizeImagesAndLandmarks((h, w), src_img, src_pts)
  dstNorm, dst_pts = fbc.normalizeImagesAndLandmarks((h, w), dst_img, dst_pts)

  # Calculate average points. Will be used for Delaunay triangulation.
  pointsAvg = (src_pts + dst_pts)/2.0

  # 8 Boundary points for Delaunay Triangulation
  boundaryPoints = fbc.getEightBoundaryPoints(h, w)
  src_pts = np.concatenate((src_pts, boundaryPoints), axis=0)
  dst_pts = np.concatenate((dst_pts, boundaryPoints), axis=0)
  pointsAvg = np.concatenate((pointsAvg, boundaryPoints), axis=0)
  # Calculate Delaunay triangulation.
  rect = (0, 0, w, h)
  dt = fbc.calculateDelaunayTriangles(rect, pointsAvg)

  # Start animation.
  alpha = 0

  while alpha < 1:
    # Compute landmark points based on morphing parameter alpha
    pointsMorph = (1 - alpha) * src_pts + alpha * dst_pts

    # Warp images such that normalized points line up with morphed points.
    imOut1 = fbc.warpImage(srcNorm, src_pts, pointsMorph.tolist(), dt)
    imOut2 = fbc.warpImage(dstNorm, dst_pts, pointsMorph.tolist(), dt)

    # Blend warped images based on morphing parameter alpha
    imMorph = (1 - alpha) * imOut1 + alpha * imOut2

    imMorph = np.uint8(imMorph * 255)
    utils.image_to_url("results/face_morph/morph_%1.2f.jpg" % alpha, imMorph)

    alpha += 0.1
  # imMorph

  return Response("")


@api_view(['POST'])
def video_morph(request):
  data = utils.parseRequest(request)
  dst_url = data['dst_img']
  src_url = data['src_img']

  src_img = utils.url_to_image(src_url)
  dst_img = utils.url_to_image(dst_url)

  utils.image_to_url("results/face_morph/morph_orig_src.jpg", src_img)
  utils.image_to_url("results/face_morph/morph_orig_dst.jpg", dst_img)

  src_pts = fbc.getLandmarks(
      faceDetector, landmarkDetector, cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))
  dst_pts = fbc.getLandmarks(
      faceDetector, landmarkDetector, cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB))

  src_pts = np.array(src_pts)
  dst_pts = np.array(dst_pts)

  # Convert image to floating point in the range 0 to 1
  src_img = np.float32(src_img)/255.0
  dst_img = np.float32(dst_img)/255.0

  h = 300
  w = 300

  # Normalize image to output coordinates.
  srcNorm, src_pts = fbc.normalizeImagesAndLandmarks((h, w), src_img, src_pts)
  dstNorm, dst_pts = fbc.normalizeImagesAndLandmarks((h, w), dst_img, dst_pts)

  # Calculate average points. Will be used for Delaunay triangulation.
  pointsAvg = (src_pts + dst_pts)/2.0

  # 8 Boundary points for Delaunay Triangulation
  boundaryPoints = fbc.getEightBoundaryPoints(h, w)
  src_pts = np.concatenate((src_pts, boundaryPoints), axis=0)
  dst_pts = np.concatenate((dst_pts, boundaryPoints), axis=0)
  pointsAvg = np.concatenate((pointsAvg, boundaryPoints), axis=0)
  # Calculate Delaunay triangulation.
  rect = (0, 0, w, h)
  dt = fbc.calculateDelaunayTriangles(rect, pointsAvg)

  # Start animation.
  alpha = 0

  frames = []

  while alpha < 1:
    # Compute landmark points based on morphing parameter alpha
    pointsMorph = (1 - alpha) * src_pts + alpha * dst_pts

    # Warp images such that normalized points line up with morphed points.
    imOut1 = fbc.warpImage(srcNorm, src_pts, pointsMorph.tolist(), dt)
    imOut2 = fbc.warpImage(dstNorm, dst_pts, pointsMorph.tolist(), dt)

    # Blend warped images based on morphing parameter alpha
    imMorph = (1 - alpha) * imOut1 + alpha * imOut2

    imMorph = np.uint8(imMorph * 255)
    utils.image_to_url("results/face_morph/morph_%1.2f.jpg" % alpha, imMorph)
    frames.append(imMorph)

    alpha += 0.05

  path = "results/face_morph/face_average.avi"
  video_utils.video_write(path, 8, (300, 300), frames)
  # video_utils.video_write("results/face_morph/face_average.mp4", 8, (300, 300), frames)
  url = utils.video_to_url(path)
  # return Response("")
  return Response({"video_url": url})