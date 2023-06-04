from django.shortcuts import render
from django.http import HttpResponse
from django.http.response import StreamingHttpResponse
import numpy as np
import cv2
import mediapipe as mp
from django.views.decorators import gzip
from signtext.camera import *
from signtext import config
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from django.http import HttpResponseNotAllowed
# from tf.keras.models import load_model
mp_holistic = mp.solutions.holistic
from django.http import HttpResponseBadRequest, JsonResponse
from django.contrib.sessions.backends.db import SessionStore
from PIL import Image
import requests

from django.core.paginator import Paginator


s = SessionStore()


module_dir = os.path.dirname(__file__)   #get current directory
model_path = os.path.join(module_dir, 'static/model_v1.h5')
labels_path = os.path.join(module_dir, 'static/labels.npy')
sequence_path = os.path.join(module_dir, 'static/max_len.npy')

data_dir = os.path.join(module_dir, 'static/data/')
gifs = os.path.join(module_dir, 'static/data/gifs/')

model = load_model(model_path)
labels = np.load(labels_path)
sequence_length = np.load(sequence_path)

sequence_length = int(sequence_length)

s['sequence'] = config.sentence
s['predictions'] = config.predictions
s['new_pred'] = config.new_pred
s['sentence'] = config.sentence
s['threshold'] = config.threshold
s['second_image'] = config.second_image
s['feedback'] = config.feedback

s['mp_holistic'] = mp.solutions.holistic
s['holistic'] = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Create your views here.

def is_ajax(request):
    return request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'

def update_session(request):
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    if s['new_pred']:
        if is_ajax:
            if request.method == 'GET':

                # response = requests.post("https://hnmensah-ghanaian-language-translator.hf.space/api/predict", json={
                #     "data": [
                #         "English",
                #         "Asante",
                #         s['new_pred'],
                # ]}).json()

                # data = str(s['new_pred']) + " - " + str(response["data"][0])
                data = str(s['new_pred'])
                
                # new_pred = data
                print(data)
                return JsonResponse({'context': data, })
            return JsonResponse({'status': 'Invalid request'}, status=400)
        else:
            return HttpResponseBadRequest('Invalid request')


def result_display_landmarks(image):

    
    with  mp_holistic.Holistic(static_image_mode=True) as holistic:
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])



def get_landmarks_predictions(request, frame):
    results = s['holistic'] .process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    landmarks =  np.concatenate([pose, face, lh, rh])
    
    
    if results.left_hand_landmarks or results.right_hand_landmarks: #check if hand shows in frame
        # print(True)
        s['feedback'] = "Sign detected"
        s['sequence'].append(landmarks) #if True, add to sequence
    
    else: # If no hands shown, check if it's end of sign or no sign at all
        if not len(s['sequence']):  # if no sign from hands
            s['feedback'] = "Please place your hand in the camera frame to begin detection"            # continue to next frame
            
         
        else:                   # if there exists other signs, means end of sign, make predictions
            s['feedback'] = "Predicting sign..."

            needed__width = sequence_length - len(s['sequence'])
            s['sequence'] = np.pad(s['sequence'], [(0, needed__width), (0,0) ], mode='constant')
            res = model.predict(np.expand_dims(s['sequence'], axis=0))
            label = labels[np.argmax(res)]
            s['predictions'].append(label)
            s['sequence'] = [] # empty sequence for next round of signs

            if len(s['new_pred']) >= 10: # prevent words overflow from screen
                new_pred = s['predictions'][::-1][:10][::-1]   # if words overflow, pick the last 5
            else:
                new_pred = s['predictions']          #else show words
            s['new_pred'] = new_pred[-1]
            # s['new_pred'] = s['predictions'][::-1][1]
            # print(s.get('new_pred', "Heloo"))
            # "Detecting sign"

                    

    
def gen(request, camera):
    while True:
        frame = camera.get_frame()
        image = camera.get_clean_frame()

        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
        # get_landmarks_predictions(request, image) #-- change this 
        request.session.modified = True


@gzip.gzip_page
def livefeed(request):
    try:
        cam = VideoCamera()
        request.session.modified = True
        return StreamingHttpResponse(gen(request, cam), content_type="multipart/x-mixed-replace;boundary=frame")
    
    except:  # This is bad!
        pass

def signtext(request):
    # request.session["new_pred"] = ' '.join(config.new_pred)
    request.session.modified = True

    context = {
        "sentence" : ' '.join(list(s['new_pred'])),
        "feedback": s['feedback'],
        's': ' '.join(s['new_pred'])
    }
    return render(request, 'signtext/signtext.html', context=context)

def practice(request):
    # request.session["new_pred"] = ' '.join(config.new_pred)
    request.session.modified = True

    context = {
        "sentence" : ' '.join(list(s['new_pred'])),
        "feedback": s['feedback'],
        's': ' '.join(s['new_pred'])
    }
    return render(request, 'signtext/practice.html', context=context)

def display_landmarks(image):
    mp_drawing = mp.solutions.drawing_utils 
    mp_drawing_styles = mp.solutions.drawing_styles

    with  mp_holistic.Holistic(static_image_mode=True) as holistic:
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            final_results = results.pose_world_landmarks.landmark
            
            mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)
            
    

def textsign(request):
    word = request.GET.get('q') if request.GET.get('q') != None else ""
    

    context = {}
    image = None

    if word != "":
        word = word.upper() + ".gif"

        print(word)
        
        
        context = {
            "word" : word
        }
    
    


    return render(request, 'signtext/textsign.html', context=context)




def index(request):
    return render(request, 'signtext/index.html')

def select_user(request):
    return render(request, 'signtext/select_user.html')

def select_solution(request):
    return render(request, 'signtext/select_solution.html')

def learn_main(request):
    categories = ["Pregnancy & Reproduction", "Emergency", "Medical conditions", "Remedies", "Medical procedures", "General : Health"]

    context = {
        "categories": categories
    }
    return render(request, 'signtext/learn_main.html', context)

def learn_page(request, pk):

    category_pair = {
        "Pregnancy & Reproduction": [["Breastfeed","Breastfeed.mp4"], ["Health","Health.mp4"], ["Labor","Labor.mp4"], ["Miscarriage","Miscarriage.mp4"], ["Breast","Breast.mp4"], ["Medicine","Medicine.mp4"], ],
        "Emergency": [["Emergency","https://youtube.com"], ["Labor","https://youtube.com"]],
        "Medical conditions": [["Conditions","https://youtube.com"], ["Labor","https://youtube.com"]],
        "Remedies": [["Remedies","https://youtube.com"], ["Labor","https://youtube.com"]],
        "Medical procedures": [["Medical","https://youtube.com"], ["Labor","https://youtube.com"]],
        "General : Health": [["Health","https://youtube.com"], ["Labor","https://youtube.com"]]
    }

    signs = category_pair[pk]

    p = Paginator(signs, 1)

    page = request.GET.get('page')
    if not page:
        page = 1

    object_list = p.page(page)

    print(len(signs))


    context = {
        "signs" : object_list,
        "percent": str( int(page) / int(p.count) * 100) + "%"
    }
    return render(request, 'signtext/learn_page.html', context)



def hospital_page(request):
    hospitals = [
        ["Komfo Anokye Teaching Hospital", "Ashanti Region", "https://www.google.com/maps/dir/6.9723244,-1.4716475/komfo+anokye+hospital/@6.8303378,-1.6584845,11z/data=!3m1!4b1!4m9!4m8!1m1!4e1!1m5!1m1!1s0xfdb96fa87e87323:0xf47a8c98b1dd0923!2m2!1d-1.6291939!2d6.6975237"],
        ["Asante Mampong Government Hospital", "Ashanti Region", ""],
        ["Korle Bu Teaching Hospital", "Greater Accra Region", ""],
        ["37 Military Hospital", "Greater Accra Region", ""],
        ["Effia Nkwanta Regional Hospital", "Western Region", ""],
        ["Essikado Hospital", "Western Region", ""],
        ["Ho Municipal Hospital", "Volta Region", ""],
        ["Bibiani Government Hospital", "Western Region", ""],

    ]

    context = {
        "hospitals" : hospitals
    }

    return render(request, 'signtext/choose_hospital.html', context)


def learn(request):
    return render(request, 'signtext/learn.html')

def community(request):
    return render(request, 'signtext/community.html')

def community_chat(request):
    return render(request, 'signtext/community_chat.html')




