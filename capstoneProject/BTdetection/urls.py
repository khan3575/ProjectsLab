from django.contrib import admin
from django.urls import include, path
from . import views

urlpatterns = [
    path('hello/', views.hello , name='hello'),
    path('login/', views.login , name='login'),
]