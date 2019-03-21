from django.urls import include, path
from rest_framework import routers
import api_practice.api as api

router = routers.DefaultRouter()

urlpatterns = [
    path('', include(router.urls)),
    path('warmer', api.warmer),
    path('cooler', api.cooler),
    path('brighten', api.brighten),
    path('darken', api.darken),
    path('oldify', api.oldify),
    path('ig_filter', api.ig_filter),
]
