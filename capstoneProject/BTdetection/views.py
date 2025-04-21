from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader

def login(request):
    #Rendering the log in page
    template = loader.get_template('login.html')
    return HttpResponse(template.render())

def hello(request):
    #Rendering the hello page
    return HttpResponse("Hello, world!")