"""nlp1 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from home import views
urlpatterns = [
    path('index', views.index),
    path('contact', views.contact),
    path('summary_url', views.summary_url),
    path('summary_doc', views.summary_doc),
    path('summary_text', views.summary_text),
    path('text_action_page', views.summarized_text),
    path('url_action_page', views.summarized_url),
    path('doc_action_page', views.summarized_doc),
    

    # path('summary_url/', views.generate_summary_for_url, name = 'summary_url'),
    # # path('first_summary_docs/', views.generate_summary_for_doc1, name = 'summary_url'),
    # path('text_action_page/',views.generate_summary_for_text2,name ="summery_text1"),
    # path('url_action_page/',views.generate_summary_for_url2,name ="summery_text1"),
    # # path('doc_action_page/',views.generate_summary_for_doc2,name ="summery_doc1"),
    # path('summary_text/', views.generate_summary_for_text , name = 'summary_text'),
    # path('summary_docs/', views.generate_summary_for_doc , name = 'summary_docs'),
    # path("",views.basic1, name = 'basic2'),
]