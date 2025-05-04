from django.db import models

class Address(models.Model):
    address = models.CharField(max_length=255)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    country = models.CharField(max_length=100)
    zip_code = models.CharField(max_length=20)
    class Meta:
        unique_together = ('address', 'city', 'state', 'country', 'zip_code')

    def __str__(self):
        return f"{self.user.username}'s Address"

# Create your models here.
class User(models.Model):
    password = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    nid = models.IntegerField(unique=True)
    phone = models.CharField(max_length=15, unique=True)
    firstName= models.CharField(max_length=50)
    lastName = models.CharField(max_length=50)
    birthDate = models.DateField()
    address = models.ForeignKey(Address, on_delete=models.SET_NULL, null=True)

    def __str__(self):
        return self.username
    
