from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.template import loader
from .models import Address, User, PasswordResetToken
from django.contrib.auth.hashers import make_password,check_password
from django.contrib.auth.decorators import login_required
from django.urls import reverse
from django.core.mail import send_mail
from django.utils.crypto import get_random_string
from django.contrib import messages

def login(request):
    #Rendering the log in page
    if request.method == 'POST':
        email = request.POST.get('email')   
        password = request.POST.get('password')
        try:
            
            user = User.objects.get(email=email)
            #if check_password(password, user.password):
            if(password == user.password):
                print("Password is correct")
                return redirect('home')  
            else:
                message = "Invalid password."
        except User.DoesNotExist:
            message = "User does not exist."
        return render(request, 'login.html', {'message': message})

    return render(request, 'login.html')

def register(request):
    if request.method == 'POST':
        
        email = request.POST.get('email')
        password = request.POST.get('password')
        #password = make_password(password)  # Hash the password
        nid = request.POST.get('nid')
        phone = request.POST.get('phone')
        first_name = request.POST.get('fname')
        last_name = request.POST.get('lname')
        birth_date = request.POST.get('bdate')

      
        #address_line = request.POST.get('address')
        city = request.POST.get('city')
        state = request.POST.get('state')
        country = request.POST.get('country')
        zip_code = request.POST.get('zip')

        address, created = Address.objects.get_or_create(
            city=city,
            state=state,
            country=country,
            zip_code=zip_code,
        )
        try:
            user = User.objects.create(
                email=email,
                password=password,
                nid=nid,
                phone=phone,
                firstName=first_name,
                lastName=last_name,
                birthDate=birth_date,
                address=address
            )
            user.save()
            message = "User registered successfully!"
            return render(request, 'login.html', {'message': message})
        except Exception as e:
            message = f"Error: {str(e)}"
            return render(request, 'registration.html', {'message': message})
        # If the user is created successfully, redirect to the login page or show a success message

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
def forgot_password(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        user = User.objects.filter(email=email).first()
        if user:
            token = get_random_string(length=32)
            # Save the token (assuming your User model has a 'reset_token' field)
            PasswordResetToken.objects.create(user=user, token=token)
            
            reset_link = request.build_absolute_uri(reverse('reset-password', args=[token]))
            send_mail(
                subject='Password Reset - GNN Assistant',
                message=f'Hi {user.firstName},\nClick the link below to reset your password:\n{reset_link}',
                from_email='no-reply@gnnassistant.com',
                recipient_list=[email],
            )
        
        # For security, always show the same message regardless of whether the user exists.
        messages.success(request, 'If the email exists, a reset link has been sent.')
        return redirect('forgot_password')
    
    return render(request, 'forgot.html')

def reset_password(request, token):
    # Find the user by token. Ensure that the token exists.
    token_obj = PasswordResetToken.objects.filter(token=token).first()
    if not token_obj:
        messages.error(request, "Invalid or expired reset token.")
        return redirect('forgot_password')
    
    user = token_obj.user
    
    if request.method == 'POST':
        new_password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        
        if new_password != confirm_password:
            messages.error(request, "Passwords do not match!")
        else:
            # Uncomment make_password() if you want to hash the new password.
            user.password = new_password  # or use: make_password(new_password)
            # Clear the token so it cannot be used again.
            user.save()
            token_obj.delete()
            messages.success(request, "Your password has been reset. You can now log in.")
            return redirect('login')
            
    return render(request, 'reset.html', {'token': token})