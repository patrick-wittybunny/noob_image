import cv2
from rest_framework import status
from rest_framework.response import Response

def adjustTone(request):
    return Response('', status=status.HTTP_200_OK)

def adjustBrightness(request):
    return 1;

def noirify(request):
    return 1;

def ig_filter(request):
    return 1;
