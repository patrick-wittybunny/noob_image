import cv2
import numpy as np
import api_practice.faceBlendCommon as fbc


def loadImageYcb(image):
    ycbImage = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycbImage = np.float32(ycbImage)
    return ycbImage


def colorDodge(top, bottom):

  # divid the bottom by inverted top image and scale back to 250
  output = cv2.divide(bottom, 255 - top, scale=256)

  return output

def sketchPencilUsingBlending(original, kernelSize=21):
  img = np.copy(original)

  # Convert to grayscale
  imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Invert the grayscale image
  imgGrayInv = 255 - imgGray

  # Apply GaussianBlur
  imgGrayInvBlur = cv2.GaussianBlur(imgGrayInv, (kernelSize, kernelSize), 0)

  # blend using color dodge
  output = colorDodge(imgGrayInvBlur, imgGray)

  return cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

def color_transfer(src, dst):

  output = np.copy(dst)
  srcLab = np.float32(cv2.cvtColor(src, cv2.COLOR_BGR2LAB))
  dstLab = np.float32(cv2.cvtColor(dst, cv2.COLOR_BGR2LAB))
  outputLab = np.float32(cv2.cvtColor(output, cv2.COLOR_BGR2LAB))

  print(src)

  print(srcLab)

  # Split the Lab images into their channels
  srcL, srcA, srcB = cv2.split(srcLab)
  dstL, dstA, dstB = cv2.split(dstLab)
  outL, outA, outB = cv2.split(outputLab)
  

  outL = dstL - dstL.mean()
  outA = dstA - dstA.mean()
  outB = dstB - dstB.mean()

  # scale the standard deviation of the destination image
  outL *= srcL.std() / (dstL.std() if dstL.std() else 1)
  outA *= srcA.std() / (dstA.std() if dstA.std() else 1)
  outB *= srcB.std() / (dstB.std() if dstB.std() else 1)

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
  #  cv2.imwrite("results/oldify_%d.jpg" % (random.randint(0, 10000)), output)
  return output

def alphablend(background, foreground):

  b, g, r, a = cv2.split(foreground)

  # Save the foregroung RGB content into a single object
  foreground = cv2.merge((b, g, r))

  # Save the alpha information into a single Mat
  alpha = cv2.merge((a, a, a))

  # Read background image
  background = cv2.imread("../data/images/backGroundLarge.jpg")

  # Convert uint8 to float
  foreground = foreground.astype(float)
  background = background.astype(float)
  alpha = alpha.astype(float)/255

  # Perform alpha blending
  foreground = cv2.multiply(alpha, foreground)
  background = cv2.multiply(1.0 - alpha, background)
  outImage = cv2.add(foreground, background)
  return outImage

def face_swap(img1, img2, points1, points2):
  
  img1Warped = np.copy(img2)

  # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
  # Find convex hull
  hull1 = []
  hull2 = []

  hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

  for i in range(0, len(hullIndex)):
    hull1.append(points1[hullIndex[i][0]])
    hull2.append(points2[hullIndex[i][0]])

  # Find delanauy traingulation for convex hull points
  sizeImg2 = img2.shape
  rect = (0, 0, sizeImg2[1], sizeImg2[0])

  dt = fbc.calculateDelaunayTriangles(rect, hull2)

  if len(dt) == 0:
    quit()

  # Apply affine transformation to Delaunay triangles
  for i in range(0, len(dt)):
    t1 = []
    t2 = []

    #get points for img1, img2 corresponding to the triangles
    for j in range(0, 3):
      t1.append(hull1[dt[i][j]])
      t2.append(hull2[dt[i][j]])

    fbc.warpTriangle(img1, img1Warped, t1, t2)

  # Calculate Mask for Seamless cloning
  hull8U = []
  for i in range(0, len(hull2)):
    hull8U.append((hull2[i][0], hull2[i][1]))

  mask = np.zeros(img2.shape, dtype=img2.dtype)

  cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
  # find center of the mask to be cloned with the destination image
  r = cv2.boundingRect(np.float32([hull2]))

  center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))

  # img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

  # Clone seamlessly.
  img2 = cv2.seamlessClone(
      np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)
    
  return img2

def face_average(img1, img2, points1, points2):
  images = []
  allPoints = []
  allPoints.append(points1)
  allPoints.append(points2)
  images.append(np.float32(img1)/255.0)
  images.append(np.float32(img2)/255.0)

  w = 300
  h = 300

  boundaryPts = fbc.getEightBoundaryPoints(h, w)

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
  # print(output)
  return output
