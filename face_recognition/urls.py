from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('effective-point/', include('effective_point.urls')),
    path('train/', include('train.urls')),
    path('recognize/', include('recognize.urls')),
]
