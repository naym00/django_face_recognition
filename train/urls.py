from django.urls import path
from train.views import train_yourself

urlpatterns = [
    path('<str:name>/<int:camera>/', train_yourself, name='train-yourself'),
]
