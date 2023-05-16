from django.urls import path

from . import views

urlpatterns = [
    path('home', views.index, name='home'),
    path('select_user', views.select_user, name='select_user'),
    path('select_solution', views.select_solution, name='select_solution'),
    path('learn_main', views.learn_main, name='learn_main'),
    path('learn_page/<str:pk>/', views.learn_page, name='learn_page'),

    path('textsign', views.textsign, name='textsign'),
    path('signtext', views.signtext, name='signtext'),
    path('learn', views.learn, name='learn'),
    path('practice', views.practice, name='practice'),
    path('live_feed', views.livefeed, name="live_feed"),
    path('update_session', views.update_session, name="update_session"),
]

