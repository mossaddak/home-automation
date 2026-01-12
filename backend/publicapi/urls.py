from django.contrib import admin
from django.urls import path, include

from .views import ProcessData

urlpatterns = [
    path(r'/process', ProcessData, name='process_data'),
]
