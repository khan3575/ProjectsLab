from django.shortcuts import render,redirect,get_object_or_404
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
            user.password = new_password  
            user.save()
            token_obj.delete()
            messages.success(request, "Your password has been reset. You can now log in.")
            return redirect('login')
            
    return render(request, 'reset.html', {'token': token})

@login_required
def user_profile(request, email):
    current_user = get_current_user(request)
    
    if not current_user:
        messages.warning(request, 'Please log in to view your profile.')
        return redirect('login')
    
    if request.method == 'POST':
        # Handle profile updates
        try:
            current_user.firstName = request.POST.get('first_name', '').strip()
            current_user.lastName = request.POST.get('last_name', '').strip()
            current_user.phone = request.POST.get('phone', '').strip()
            
            # Update address if provided
            city = request.POST.get('city', '').strip()
            state = request.POST.get('state', '').strip()
            country = request.POST.get('country', '').strip()
            zip_code = request.POST.get('zip', '').strip()
            
            if city and state and country:
                address, created = Address.objects.get_or_create(
                    city=city,
                    state=state,
                    country=country,
                    zip_code=zip_code
                )
                current_user.address = address
            
            current_user.save()
            messages.success(request, 'Profile updated successfully!')
            return redirect('profile')
            
        except Exception as e:
            messages.error(request, f'Error updating profile: {str(e)}')
    
    context = {
        'user_profile': current_user,
        'current_user': current_user,
    }
    return render(request, 'profile.html', context)


from django.core.mail import send_mail

def contact(request):
    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        subject = request.POST.get("subject")
        message_content = request.POST.get("message")
        
        full_message = f"Message from {name} ({email}):\n\n{message_content}"
        
        try:
            send_mail(
                subject,                             # Email subject
                full_message,                        # Email body/message
                "no-reply@gnnassistant.com",         # From email address
                ["contact@gnnassistant.com"],        # Recipient list (adjust accordingly)
            )
            messages.success(request, "Thank you for your message. We will get back to you soon.")
        except Exception as e:
            messages.error(request, "There was an error sending your message. Please try again later.")
        
        return redirect("contact")
    
    return render(request, "contact.html")

def get_current_user(request):
    user_id = request.session.get('user_id')
    if user_id:
        try:
            return User.objects.get(id=user_id)
        except User.DoesNotExist:
            return None
    return None

def logout(request):
    current_user = get_current_user(request)
    if current_user:
        request.session.flush()
        messages.success(request, 'You have been logged out successfully.')
    return redirect('home')