from django.urls import path

from . import views

urlpatterns = [
    path('home', views.index, name='home'),
    path('textsign', views.textsign, name='textsign'),
    path('signtext', views.signtext, name='signtext'),
    path('learn', views.learn, name='learn'),
    path('practice', views.practice, name='practice'),
    path('live_feed', views.livefeed, name="live_feed"),
    path('update_session', views.update_session, name="update_session"),
]

