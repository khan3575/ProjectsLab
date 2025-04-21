from django.contrib import admin
from django.urls import include, path
from . import views

urlpatterns = [
    path('login/', views.login , name='login'),
    path('register/', views.register , name='register'),
    path('about/', views.about , name='about'),
    path('upload/', views.upload , name='upload'),
    path('history/', views.history , name='history'),
    path('', views.home , name='home'),

]