from django.urls import include, path
from rest_framework import routers
from . import week1

router = routers.DefaultRouter()

urlpatterns = [
    path('', include(router.urls)),
    path(r'/adjustTone', week1.adjustTone),
    path(r'/adjustBrightness', week1.adjustTone),
    path(r'/noirify', week1.adjustTone),
    path(r'/ig_filter', week1.adjustTone),
]

