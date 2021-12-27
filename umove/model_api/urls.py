from django.urls import path, include
from . import views
from rest_framework import routers
from django.conf.urls import url

router = routers.DefaultRouter()

urlpatterns = [
    # Path for running model
    path('run_model/', views.run_model, name='run_model'),
]