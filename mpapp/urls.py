from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('simplex/', views.simplex_view, name='simplex'),
    path('graphical/', views.lp_solver_view, name='graphical'),
]

