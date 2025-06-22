from django.urls import path
from . import views

urlpatterns = [
    path('score_each_hour/', views.score_each_hour_api, name='score_each_hour_api'),
]