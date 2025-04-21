from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader

def login(request):
    #Rendering the log in page
    template = loader.get_template('login.html')
    return HttpResponse(template.render())

def register(request):
    return render(request, 'registration.html')

def about(request):
    #Rendering the about page
    return render(request, 'about.html')

def upload(request):
    #Rendering the upload page
    return render(request, 'upload.html')
def history(request):
    #Rendering the history page
    return render(request, 'history.html')
def home(request):
    #Rendering the home page
    return render(request, 'home.html')