from django.db import models
from django.utils import timezone

class Address(models.Model):
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    country = models.CharField(max_length=100)
    zip_code = models.CharField(max_length=20)
    
    class Meta:
        unique_together = ('city', 'state', 'country', 'zip_code')

    def __str__(self):
        return f"{self.address or ''}, {self.city}, {self.state}, {self.country}, {self.zip_code}"

# Create your models here.

class User(models.Model):
    
    password = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    nid = models.IntegerField(unique=True)
    phone = models.CharField(max_length=15, unique=True)
    firstName = models.CharField(max_length=50)
    lastName = models.CharField(max_length=50)
    birthDate = models.DateField()
    address = models.ForeignKey(Address, on_delete=models.SET_NULL, null=True)
    
    def __str__(self):
        return f"{self.firstName} {self.lastName} ({self.email})"
    
class PasswordResetToken(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    token = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.user.email} - {self.token}"
    

class scan(models.Model):
    user_id = models.ForeignKey(User, on_delete=models.CASCADE)
    fileName = models.CharField(max_length=255)
    scan_file = models.FileField(upload_to='scans/', default='default.nii')

class prediction(models.Model):
       scan_id = models.ForeignKey(scan, on_delete=models.CASCADE, related_name='prediction')
       result_data = models.FloatField()
       confidence = models.FloatField()
       predicted_at = models.DateField()
       result = models.CharField(max_length=20, default='Pending')  # Default specified here

       def __str__(self):
           return f"{self.scan_id} - {self.result}"