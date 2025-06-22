from django.urls import path
from . import views

urlpatterns = [
    path('score_each_hour/', views.score_each_hour_api, name='score_each_hour_api'),
    path('test/', views.test_get_all_trails, name='test_trails'),
    path('upload/', views.upload_file, name='upload_file'),
]