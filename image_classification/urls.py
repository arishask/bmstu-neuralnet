from django.urls import path
from . import views

app_name = 'image_classification'
urlpatterns = [
    path('', views.index, name='index'),
    path('<int:task_id>/', views.detail, name='detail'),
    path('<int:task_id>/predict/', views.predict, name='predict'),
]
