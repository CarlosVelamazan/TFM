from django.conf.urls import include, url
from . import views

urlpatterns = [
    url(r'^$', views.main, name='home'),
    url(r'^cluster/detail/', views.cluster_detail, name='cluster_detail'),
    url(r'^cluster/new/$', views.search, name='search'),
]
