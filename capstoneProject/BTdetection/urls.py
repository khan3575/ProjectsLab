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
    path('forgot_password/', views.forgot_password , name='forgot_password'),
    path('reset/<str:token>/', views.reset_password, name='reset-password'),
    path('profile/', views.profile, name='profile'),
    path('contact/', views.contact, name='contact'),
    path('logout/', views.logout , name='logout' ),
    path('predict/', views.predict_view, name='predict'),
    path('results/', views.results, name='results'),
  
]