from django.urls import path
from effective_point.views import effective_points

urlpatterns = [
    path('', effective_points, name='effective-points'),
]
