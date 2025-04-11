from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
               path("predictModel.html", views.predictModel, name="predictModel"),	      
               path("predictModelAction", views.predictModelAction, name="predictModelAction"),
               path("trainModel", views.trainModel, name="trainModel"),               	       
]
