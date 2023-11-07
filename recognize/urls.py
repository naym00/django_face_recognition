from django.urls import path
from recognize.views import recognize_yourself

urlpatterns = [
    path('<int:camera>/', recognize_yourself, name='recognize-yourself'),
    
]
