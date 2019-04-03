# noob_image

How to run server?

1. Enter your virtual environment (if any)
2. pip install django djangorestframework
3. python manage.py runserver

This turns on a default server on 127.0.0.1:8000. To run it in a different address or port, you can specify it after run server

```
python manage.py runserver <port/address>
```


To add your own functions, create it in any file (in this project, functions are in api.py or in api2.py), then decorate it
```
# assuming this function is defined in functions.py
@api_view(['POST']) #decorator
def myFunction(request):
  return Reponse('')
```
Then add it to urls.py
```
#api_practice is the name space of this current project, we import the file
import api_practice.functions as functions
...
...
...
urlpatterns = [
    ...,
    # we specify the end point that we want to point to the function
    path('/myFunc1', functions.myFunction),
]
```

Once you request to 127.0.0.1:8000/myFunc1, it will forward that request to the function you have written. To unwrap the request, you can simply call utils.parseRequest(request) and it will return a dictionary format of the json body of the request
```
import api_practice.utils as utils
...
...
@api_view(['POST'])
def myFunction(request):
  body = utils.parseRequest(request)
  print(body)
  if body['name'] is not None:
    return Reponse({"name": body['name']})
  else:
    return Response(404)
```


P.S.
1. Please use cloudinary sparingly as it is only for free and has maximum storage memory limit.
In utils.py, comment out uploading and removal of local file
```
def image_to_url(path, image):
  cv2.imwrite(path, image)
  #a = cloudinary.uploader.upload(path)
  #os.remove(path)
  #return a['url']
  return ""
  
def video_to_url(path):
  #a = cloudinary.uploader.upload(path, resource_type = "video")
  #os.remove(path)
  #return a['url']
  return ""
```
