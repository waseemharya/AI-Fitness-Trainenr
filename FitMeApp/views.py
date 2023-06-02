from django.conf import settings
from django.core.mail import send_mail
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from user.models import *
from .forms import *
from .models import *
from .models import Videos
import cv2
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
import threading
import pandas as pd
import sqlite3
import mediapipe as mp
import numpy as np
import streamlit as st
import os
##########################################################################################################

# Connect to the SQLite database
conn = sqlite3.connect('db.sqlite3')
# Read the table into a pandas DataFrame
df = pd.read_sql_query('SELECT * FROM FitMeApp_videos', conn)
# Get the last row of the DataFrame
last_entry = df.tail(1)
# Select a field from the last row
last_field_value = last_entry.iloc[0]['video']
#mm = "/home/atif/Desktop/4.mp4" #("/home/atif/Desktop/django website gym/media/" + last_field_value) 
from FitMe.settings import BASE_DIR

mm = (BASE_DIR + "/media/" + last_field_value)
print(mm)
print("/home/atif/Desktop/gym-web/media/" + last_field_value)
# Close the database connection
conn.close()
###################################################################

def home(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(username=username, password=password)
        if user is not None:
            try:
                trainer = Trainer.objects.get(user=user)
            except:
                trainer = False
            if trainer:
                if trainer.approve:
                    login(request, user)
                    messages.success(request, f" welcome {username} !!")
                    try:
                        trainee = Trainee.objects.filter(trainer_ass=trainer)
                    except:
                        trainee = None
                    print("---------------------------------")
                    data = {"trainee": trainee}
                    return render(request, "TrainerDashBoard.html", data)
                else:
                    messages.success(
                        request, f" wecome {username} please ask admin to approve !!"
                    )
            else:
                login(request, user)
                try:
                    task = Task.objects.filter(
                        person=Trainee.objects.get(user=request.user)
                    )
                except:
                    task = None

                data = {"task": task}
                messages.success(request, f" welcome {username} !!")
                return render(request, "first.html")

        else:
            messages.info(request, f"account done not exit plz sign in")
    if request.user.is_anonymous:
        form = AuthenticationForm()
        return render(request, "index.html", {"form": form})
    else:
        try:
            trainer = Trainer.objects.get(user=request.user)
        except:
            trainer = False
        if trainer:
            if trainer.approve:
                try:
                    trainee = Trainee.objects.filter(trainer_ass=trainer)
                except:
                    trainee = None

                data = {"trainee": trainee}
                return render(request, "TrainerDashBoard.html", data)
            else:
                messages.success(
                    request, f" welcome {username} please ask admin to approve !!"
                )
        else:
            try:
                task = Task.objects.filter(
                    person=Trainee.objects.get(user=request.user)
                )
            except:
                task = None
            data = {"task": task}
            return render(request, "first.html", data)


######################################################################################################
@login_required
def giveTask(request, username):
    if request.method == "POST":
        form = TaskForm(request.POST)
        print(form.errors)
        if form.is_valid():
            trainee_here = Trainee.objects.get(user__username=username)
            note = request.POST["note"]
            task_to_give = request.POST["task_to_give"]
            Task.objects.create(
                person=trainee_here, note=note, task_to_give=task_to_give
            )
            messages.success(request, f"task given to user")

    form = TaskForm()
    data = {"form": form, "username": username}
    return render(request, "task.html", data)


@login_required
def seetask(request):
    data = {"task": Task.objects.filter(person=Trainee.objects.get(user=request.user))}
    return render(request, "seetask.html", data)


@login_required
def doneTask(request, id):
    task = Task.objects.get(id=id)
    task.task_complete = True
    task.save()
    print(Task.objects.get(id=id).task_complete)
    return redirect("seetask")


def about(request):
    return render(request, "about.html")


def portal(request):
    return render(request, "first.html")


def beginners_routines(request):
    return render(request, "beginners_routines.html")


def beginner_day1(request):
    return render(request, "beginner_day1.html")


def beginner_day2(request):
    return render(request, "beginner_day2.html")


def beginner_day3(request):
    return render(request, "beginner_day3.html")


def beginner_day4(request):
    return render(request, "beginner_day4.html")


def beginner_day5(request):
    return render(request, "beginner_day5.html")


def beginner_day6(request):
    return render(request, "beginner_day2.html")


def beginner_day7(request):
    return render(request, "beginner_day3.html")


def beginner_day8(request):
    return render(request, "beginner_day4.html")


def beginner_day9(request):
    return render(request, "beginner_day9.html")


def beginner_day10(request):
    return render(request, "beginner_day10.html")


def beginner_day11(request):
    return render(request, "beginner_day11.html")


def beginner_day12(request):
    return render(request, "beginner_day12.html")


def beginner_day13(request):
    return render(request, "beginner_day13.html")


def beginner_day14(request):
    return render(request, "beginner_day14.html")


def beginner_day15(request):
    return render(request, "beginner_day15.html")


def beginner_day16(request):
    return render(request, "beginner_day16.html")


def beginner_day17(request):
    return render(request, "beginner_day17.html")


def beginner_day18(request):
    return render(request, "beginner_day18.html")


def beginner_day19(request):
    return render(request, "beginner_day19.html")


def beginner_day20(request):
    return render(request, "beginner_day20.html")


def beginner_day21(request):
    return render(request, "beginner_day21.html")


def beginner_day22(request):
    return render(request, "beginner_day22.html")


def beginner_day23(request):
    return render(request, "beginner_day23.html")


def beginner_day24(request):
    return render(request, "beginner_day24.html")


def beginner_day25(request):
    return render(request, "beginner_day25.html")


def beginner_day26(request):
    return render(request, "beginner_day26.html")


def beginner_day27(request):
    return render(request, "beginner_day27.html")


def beginner_day28(request):
    return render(request, "beginner_day28.html")


def diet_beginner(request):
    return render(request, "diet_beginner.html")


def diet_intermediate(request):
    return render(request, "diet_intermediate.html")


def diet_hardcore(request):
    return render(request, "diet_hardcore.html")


def services(request):
    return render(request, "services.html")


def gallery(request):
    return render(request, "gallery.html")


def contact(request):
    return render(request, "contact.html")


def bmimetric(request):
    return render(request, "Fit.html")


def bmistandard(request):
    return render(request, "Standard.html")

def video(request):
    return render(request, "video.html")

def gym(request):
    return render(request, "gym.html")



def camera(request): #Excerise: 1. Bicep_curl 
    mp_drawing=mp.solutions.drawing_utils
    mp_pose=mp.solutions.pose

    #'Excerise: 1. Bicep_curl 2. Shoulder_press 3. Highknees'
    if 5 ==5:    
        def calculate_angle(a,b,c):
            a = np.array(a)  #first angle
            b = np.array(b)  #second angle
            c = np.array(c)  #third angle
            radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle>180.0:
                angle = 360-angle
            return angle
        #Curl Counter
        cap = cv2.VideoCapture(0)
        counter = 0
        stage = None
        error=0
        ## Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                #Selection of Excercise
                try:
                    landmarks = results.pose_landmarks.landmark
                    # Get coordinates
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    # Calculate angle
                    angle_1 =calculate_angle(left_shoulder, left_elbow, left_wrist)
                    angle_2=calculate_angle(right_shoulder,right_elbow,right_wrist)

                    # Visualize angle
                    cv2.putText(image, str(angle_1), 
                                tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )

                    cv2.putText(image, str(angle_2), 
                                tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )

                    #CURL COUNTER LOGIC
                    if angle_1 > 160 and angle_2 >160 :
                        stage = "Down"
                    if angle_1 < 30 and angle_2 < 30 and stage == 'Down':
                        stage = "up"
                        counter +=1
                        print(counter)

                except:
                    pass

                #setting up curl counter box
                cv2.rectangle(image, (0,0), (360,72), (245,117,16), -1)

                #sending values to curl counter box
                cv2.putText(image, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), 
                            (10,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

                #printing hand stage while exercising

                cv2.putText(image, 'STAGE', (165,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, 
                            (165,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

                cv2.putText(image, 'DOUBLE BICEP CURLS', (390,18), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (137, 207, 240), 1, cv2.LINE_AA)

                if counter >=1 and counter <2:
                            cv2.putText(image, '1 Set completed ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

                if counter >=2 and counter <3:
                            cv2.putText(image, '2 Set completed ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

                if counter ==3 :
                            cv2.putText(image, '3 Set completed ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

                if counter >3 :
                            cv2.rectangle(image, (40,200), (600,72), (255,255,255), -1)


                            cv2.putText(image, 'Please Stop Workout!!!', (60,150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 1, cv2.LINE_AA)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        )
                #setting up curl counter box
                cv2.rectangle(image, (0,0), (360,72), (245,117,16), -1)

                #sending values to curl counter box
                cv2.putText(image, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), 
                            (10,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                #printing hand stage while exercising

                cv2.putText(image, 'STAGE', (165,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, 
                            (165,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                cv2.putText(image, 'HIGH KNEES', (540,18), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

                if counter >=1 and counter <2:
                            cv2.putText(image, '1 Set completed ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter >=2 and counter <3:
                            cv2.putText(image, '2 Set completed ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter ==3 :
                            cv2.putText(image, '3 Set completed ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter >3 :

                            cv2.rectangle(image, (40,200), (600,72), (255,255,255), -1)


                            cv2.putText(image, 'Please Stop Workout!!!', (60,150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 1, cv2.LINE_AA)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        )     
        
                cv2.imshow('Webcam Video - Start Excercise', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break        
        

    cap.release()
    cv2.destroyAllWindows()
            
        
    return render(request, "camera.html")



    

import cv2
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
import threading

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        return frame

@gzip.gzip_page
def video_feed(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(cv2.imencode('.jpg', frame)[1]) + b'\r\n\r\n')


from .forms import SignUpForm
 
def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            user.refresh_from_db()  
            # load the profile instance created by the signal
            user.save()
            raw_password = form.cleaned_data.get('password1')

            # login user after signing up
            user = authenticate(username=user.username, password=raw_password)
            login(request, user)

            # redirect user to home page
            return redirect('/home')
    else:
        form = SignUpForm()
    return render(request, 'user/register.html', {'form': form})

def camera2(request): #Excerise: 1. Bicep_curl 
    mp_drawing=mp.solutions.drawing_utils
    mp_pose=mp.solutions.pose
    #'Excerise: 1. Bicep_curl 2. Shoulder_press 3. Highknees'
    if 5 ==5:    
        choice = 2
        def calculate_angle(a,b,c):
            a = np.array(a)  #first angle
            b = np.array(b)  #second angle
            c = np.array(c)  #third angle
            
            radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle>180.0:
                angle = 360-angle
            return angle

        #Curl Counter
        cap = cv2.VideoCapture(0)
        counter = 0
        stage = None
        error=0
        ## Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                #Selection of Excercise
                if(choice==1):
                    try:
                        landmarks = results.pose_landmarks.landmark

                        # Get coordinates
                        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                        # Calculate angle
                        angle_1 =calculate_angle(left_shoulder, left_elbow, left_wrist)
                        angle_2=calculate_angle(right_shoulder,right_elbow,right_wrist)

                        # Visualize angle
                        cv2.putText(image, str(angle_1), 
                                    tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                            )

                        cv2.putText(image, str(angle_2), 
                                    tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                            )

                        #CURL COUNTER LOGIC
                        if angle_1 > 160 and angle_2 >160 :
                            stage = "Down"
                        if angle_1 < 30 and angle_2 < 30 and stage == 'Down':
                            stage = "up"
                            counter +=1
                            print(counter)

                    except:
                        pass

                    #setting up curl counter box
                    cv2.rectangle(image, (0,0), (360,72), (245,117,16), -1)

                    #sending values to curl counter box
                    cv2.putText(image, 'REPS', (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(counter), 
                                (10,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

                    #printing hand stage while exercising

                    cv2.putText(image, 'STAGE', (165,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, stage, 
                                (165,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

                    cv2.putText(image, 'DOUBLE BICEP CURLS', (390,18), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (137, 207, 240), 1, cv2.LINE_AA)

                    if counter >=1 and counter <2:
                                cv2.putText(image, '1 Set completed ', (390,90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

                    if counter >=2 and counter <3:
                                cv2.putText(image, '2 Set completed ', (390,90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

                    if counter ==3 :
                                cv2.putText(image, '3 Set completed ', (390,90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

                    if counter >3 :
                                cv2.rectangle(image, (40,200), (600,72), (255,255,255), -1)


                                cv2.putText(image, 'Please Stop Workout!!!', (60,150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 1, cv2.LINE_AA)

                    # Render detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                            )
                elif(choice==2):
                        try:
                            landmarks = results.pose_landmarks.landmark

                            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                            # Calculate angle
                            angle_1 = calculate_angle(left_shoulder, left_elbow, left_wrist)
                            angle_2=calculate_angle(right_shoulder,right_elbow,right_wrist)

                            # Visualize angle
                            cv2.putText(image, str(angle_1), 
                                        tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                                )

                            cv2.putText(image, str(angle_2), 
                                        tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                                )

                            #CURL COUNTER LOGIC
                            if angle_1 > 110 and angle_2 >110 :
                                stage = "Up"
                            if angle_1 < 90 and angle_2 < 90 and stage == 'Up':
                                stage = "Down"
                                counter +=1
                                print(counter)


                        except:
                            pass

                        #setting up curl counter box
                        cv2.rectangle(image, (0,0), (360,72), (245,117,16), -1)

                        #sending values to curl counter box
                        cv2.putText(image, 'REPS', (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(counter), 
                                    (10,60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                        #printing hand stage while exercising

                        cv2.putText(image, 'STAGE', (165,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        cv2.putText(image, stage, 
                                    (165,60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                        cv2.putText(image, 'SHOULDER_PRESS', (390,18), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                        if counter >=1 and counter <2:
                                    cv2.putText(image, '1 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

                        if counter >=2 and counter <3:
                                    cv2.putText(image, '2 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

                        if counter ==3 :
                                    cv2.putText(image, '3 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

                        if counter >3 :

                                    cv2.rectangle(image, (40,200), (600,72), (255,255,255), -1)


                                    cv2.putText(image, 'Please Stop Workout!!!', (60,150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 1, cv2.LINE_AA)

                        # Render detections
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                                )            
                
                elif(choice==3):
                        
                        # Extract landmarks
                        try:
                            landmarks = results.pose_landmarks.landmark

                            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                            # Calculate angle
                            angle = calculate_angle(shoulder, hip, knee)

                            # Visualize angle
                            cv2.putText(image, str(angle), 
                                        tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                                )

                            #CURL COUNTER LOGIC
                            if angle > 160:
                                stage = "Down"
                            if angle < 80 and stage == 'Down':
                                stage = "Up"
                                counter +=1
                                print(counter)


                        except:
                            pass

                        #setting up curl counter box
                        cv2.rectangle(image, (0,0), (360,72), (245,117,16), -1)

                        #sending values to curl counter box
                        cv2.putText(image, 'REPS', (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(counter), 
                                    (10,60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                        #printing hand stage while exercising

                        cv2.putText(image, 'STAGE', (165,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        cv2.putText(image, stage, 
                                    (165,60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                        cv2.putText(image, 'HIGH KNEES', (540,18), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

                        if counter >=1 and counter <2:
                                    cv2.putText(image, '1 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                        if counter >=2 and counter <3:
                                    cv2.putText(image, '2 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                        if counter ==3 :
                                    cv2.putText(image, '3 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                        if counter >3 :

                                    cv2.rectangle(image, (40,200), (600,72), (255,255,255), -1)


                                    cv2.putText(image, 'Please Stop Workout!!!', (60,150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 1, cv2.LINE_AA)

                        # Render detections
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                                )     
                else:
                    print("Invalid choice!!!")
                    break
                
                cv2.imshow('Webcam Video - Start Excercise', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                
            

            cap.release()
            cv2.destroyAllWindows()
    return render(request, 'camera.html')       

def camera3(request): #Excerise: 1. Bicep_curl 
    mp_drawing=mp.solutions.drawing_utils
    mp_pose=mp.solutions.pose
    #'Excerise: 1. Bicep_curl 2. Shoulder_press 3. Highknees'
    if 5 ==5:    
        choice = 3
        def calculate_angle(a,b,c):
            a = np.array(a)  #first angle
            b = np.array(b)  #second angle
            c = np.array(c)  #third angle
            
            radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle>180.0:
                angle = 360-angle
            return angle

        #Curl Counter
        cap = cv2.VideoCapture(0)
        counter = 0
        stage = None
        error=0
        ## Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                #Selection of Excercise
                if(choice==1):
                    try:
                        landmarks = results.pose_landmarks.landmark

                        # Get coordinates
                        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                        # Calculate angle
                        angle_1 =calculate_angle(left_shoulder, left_elbow, left_wrist)
                        angle_2=calculate_angle(right_shoulder,right_elbow,right_wrist)

                        # Visualize angle
                        cv2.putText(image, str(angle_1), 
                                    tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                            )

                        cv2.putText(image, str(angle_2), 
                                    tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                            )

                        #CURL COUNTER LOGIC
                        if angle_1 > 160 and angle_2 >160 :
                            stage = "Down"
                        if angle_1 < 30 and angle_2 < 30 and stage == 'Down':
                            stage = "up"
                            counter +=1
                            print(counter)

                    except:
                        pass

                    #setting up curl counter box
                    cv2.rectangle(image, (0,0), (360,72), (245,117,16), -1)

                    #sending values to curl counter box
                    cv2.putText(image, 'REPS', (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(counter), 
                                (10,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

                    #printing hand stage while exercising

                    cv2.putText(image, 'STAGE', (165,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, stage, 
                                (165,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

                    cv2.putText(image, 'DOUBLE BICEP CURLS', (390,18), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (137, 207, 240), 1, cv2.LINE_AA)

                    if counter >=1 and counter <2:
                                cv2.putText(image, '1 Set completed ', (390,90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

                    if counter >=2 and counter <3:
                                cv2.putText(image, '2 Set completed ', (390,90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

                    if counter ==3 :
                                cv2.putText(image, '3 Set completed ', (390,90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

                    if counter >3 :
                                cv2.rectangle(image, (40,200), (600,72), (255,255,255), -1)


                                cv2.putText(image, 'Please Stop Workout!!!', (60,150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 1, cv2.LINE_AA)

                    # Render detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                            )
                elif(choice==2):
                        try:
                            landmarks = results.pose_landmarks.landmark

                            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                            # Calculate angle
                            angle_1 = calculate_angle(left_shoulder, left_elbow, left_wrist)
                            angle_2=calculate_angle(right_shoulder,right_elbow,right_wrist)

                            # Visualize angle
                            cv2.putText(image, str(angle_1), 
                                        tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                                )

                            cv2.putText(image, str(angle_2), 
                                        tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                                )

                            #CURL COUNTER LOGIC
                            if angle_1 > 110 and angle_2 >110 :
                                stage = "Up"
                            if angle_1 < 90 and angle_2 < 90 and stage == 'Up':
                                stage = "Down"
                                counter +=1
                                print(counter)


                        except:
                            pass

                        #setting up curl counter box
                        cv2.rectangle(image, (0,0), (360,72), (245,117,16), -1)

                        #sending values to curl counter box
                        cv2.putText(image, 'REPS', (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(counter), 
                                    (10,60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                        #printing hand stage while exercising

                        cv2.putText(image, 'STAGE', (165,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        cv2.putText(image, stage, 
                                    (165,60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                        cv2.putText(image, 'SHOULDER_PRESS', (390,18), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                        if counter >=1 and counter <2:
                                    cv2.putText(image, '1 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

                        if counter >=2 and counter <3:
                                    cv2.putText(image, '2 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

                        if counter ==3 :
                                    cv2.putText(image, '3 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

                        if counter >3 :

                                    cv2.rectangle(image, (40,200), (600,72), (255,255,255), -1)


                                    cv2.putText(image, 'Please Stop Workout!!!', (60,150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 1, cv2.LINE_AA)

                        # Render detections
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                                )            
                
                elif(choice==3):
                        
                        # Extract landmarks
                        try:
                            landmarks = results.pose_landmarks.landmark

                            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                            # Calculate angle
                            angle = calculate_angle(shoulder, hip, knee)

                            # Visualize angle
                            cv2.putText(image, str(angle), 
                                        tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                                )

                            #CURL COUNTER LOGIC
                            if angle > 160:
                                stage = "Down"
                            if angle < 80 and stage == 'Down':
                                stage = "Up"
                                counter +=1
                                print(counter)


                        except:
                            pass

                        #setting up curl counter box
                        cv2.rectangle(image, (0,0), (360,72), (245,117,16), -1)

                        #sending values to curl counter box
                        cv2.putText(image, 'REPS', (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(counter), 
                                    (10,60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                        #printing hand stage while exercising

                        cv2.putText(image, 'STAGE', (165,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        cv2.putText(image, stage, 
                                    (165,60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                        cv2.putText(image, 'HIGH KNEES', (540,18), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

                        if counter >=1 and counter <2:
                                    cv2.putText(image, '1 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                        if counter >=2 and counter <3:
                                    cv2.putText(image, '2 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                        if counter ==3 :
                                    cv2.putText(image, '3 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                        if counter >3 :

                                    cv2.rectangle(image, (40,200), (600,72), (255,255,255), -1)


                                    cv2.putText(image, 'Please Stop Workout!!!', (60,150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 1, cv2.LINE_AA)

                        # Render detections
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                                )     
                else:
                    print("Invalid choice!!!")
                    break
                
                cv2.imshow('Webcam Video - Start Excercise', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                
            

            cap.release()
            cv2.destroyAllWindows()
    return render(request, 'camera.html')  



from django.shortcuts import  render
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from .models import Videos
from .forms import VideoForm

def upload(request):

    vid= Videos.objects.all()

    form= VideoForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        form.save()


    context= {'form': form,
              'vid':vid,
              }
    
      
    return render(request, 'upload.html', context)


def exc_plans(request):
    return render(request, 'exc_plans.html')

def over(request):
    return render(request, 'over.html')
def under(request):
    return render(request, 'under.html')
def normal(request):
    return render(request, 'normal.html')   
##################################################################################################################################
def camera4(request): #Excerise: 1. Bicep_curl 
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    def calculate_angle(a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle = 360-angle
            
        return angle


    def rescale_frame(frame, percent=50):
        width = int(frame.shape[1] * percent/ 100)
        height = int(frame.shape[0] * percent/ 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


    # Youtube video
    angle_min = []
    angle_min_hip = []
    cap = cv2.VideoCapture("demo.mp4")
    # Curl counter variables
    counter = 0 
    min_ang = 0
    max_ang = 0
    min_ang_hip = 0
    max_ang_hip = 0
    stage = None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (640, 480)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('demo.mp4', fourcc, 24, size)


    cap = cv2.VideoCapture(0)
    # Curl counter variables
    counter = 0 
    stage = None

    """width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('your_video.mp4', fourcc, 10.0, size, True)"""

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                
                # Get coordinates
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
            
                
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                
                angle_knee = calculate_angle(hip, knee, ankle) #Knee joint angle
                
                angle_hip = calculate_angle(shoulder, hip, knee)
                hip_angle = 180-angle_hip
                knee_angle = 180-angle_knee
                
                # Visualize angle
                """cv2.putText(image, str(angle), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )"""
                        
                    
                cv2.putText(image, str(angle_knee), 
                            tuple(np.multiply(knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (79, 121, 66), 2, cv2.LINE_AA
                                    )
                
                """cv2.putText(image, str(angle_hip), 
                            tuple(np.multiply(hip, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )"""
                
                # Curl counter logic
                if angle_knee > 169:
                    stage = "UP"
                if angle_knee <= 90 and stage =='UP':
                    stage="DOWN"
                    counter +=1
                    print(counter)
            except:
                pass
            
            # Render squat counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (65,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (60,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            out.write(image)
            cv2.imshow('Squats Excercise', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        #out.release()
        cv2.destroyAllWindows()
            
        
    return render(request, "camera.html")




def camera5(request): #Excerise: 1. Bicep_curl 
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    def calculate_angle(a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle = 360-angle
            
        return angle


    def rescale_frame(frame, percent=50):
        width = int(frame.shape[1] * percent/ 100)
        height = int(frame.shape[0] * percent/ 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


    # Youtube video
    angle_min = []
    angle_min_hip = []
    cap = cv2.VideoCapture("demo.mp4")
    # Curl counter variables
    counter = 0 
    min_ang = 0
    max_ang = 0
    min_ang_hip = 0
    max_ang_hip = 0
    stage = None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (640, 480)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('demo.mp4', fourcc, 24, size)
    cap = cv2.VideoCapture(0)
    # Curl counter variables
    counter = 0 
    stage = None

    """width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('your_video.mp4', fourcc, 10.0, size, True)"""

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                
                # Get coordinates
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]           
                
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                
                angle_knee = calculate_angle(hip, knee, ankle) #Knee joint angle
                
                angle_hip = calculate_angle(shoulder, hip, knee)
                hip_angle = 180-angle_hip
                knee_angle = 180-angle_knee
                
                # Visualize angle
                """cv2.putText(image, str(angle), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )"""
                        
                    
                cv2.putText(image, str(angle_knee), 
                            tuple(np.multiply(knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (79, 121, 66), 2, cv2.LINE_AA
                                    )
                
                """cv2.putText(image, str(angle_hip), 
                            tuple(np.multiply(hip, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )"""
                
                # Curl counter logic
                if angle_knee > 169:
                    stage = "UP"
                if angle_knee <= 90 and stage =='UP':
                    stage="DOWN"
                    counter +=1
                    print(counter)
            except:
                pass
            
            # Render squat counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (65,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (60,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            out.write(image)
            cv2.imshow('Squats Excercise', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        #out.release()
        cv2.destroyAllWindows() 
    return render(request, "camera.html")

##########################################################################################################################################
def camera_vid(request): #Excerise: 1. Bicep_curl 
    mp_drawing=mp.solutions.drawing_utils
    mp_pose=mp.solutions.pose
    #mod = Videos.objects.values_list('video', flat=True).distinct()[0]
    mod = Videos.objects.latest('video')
    m = str(mod)
    # mm = ("/home/atif/Desktop/django website gym/media/" + last_field_value) 
    v1 =mm
    print("This is v: " + v1)
    #'Excerise: 1. Bicep_curl 2. Shoulder_press 3. Highknees'
    if 5 ==5:    
        def calculate_angle(a,b,c):
            a = np.array(a)  #first angle
            b = np.array(b)  #second angle
            c = np.array(c)  #third angle
            
            radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle>180.0:
                angle = 360-angle
            return angle

        #Curl Counter
        cap = cv2.VideoCapture(v1)
        counter = 0
        stage = None
        error=0
        ## Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                #Selection of Excercise

                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    # Calculate angle
                    angle_1 =calculate_angle(left_shoulder, left_elbow, left_wrist)
                    angle_2=calculate_angle(right_shoulder,right_elbow,right_wrist)

                    # Visualize angle
                    cv2.putText(image, str(angle_1), 
                                tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )

                    cv2.putText(image, str(angle_2), 
                                tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )

                    #CURL COUNTER LOGIC
                    if angle_1 > 160 and angle_2 >160 :
                        stage = "Down"
                    if angle_1 < 30 and angle_2 < 30 and stage == 'Down':
                        stage = "up"
                        counter +=1
                        print(counter)

                except:
                    pass

                #setting up curl counter box
                cv2.rectangle(image, (0,0), (360,72), (245,117,16), -1)

                #sending values to curl counter box
                cv2.putText(image, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), 
                            (10,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

                #printing hand stage while exercising

                cv2.putText(image, 'STAGE', (165,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, 
                            (165,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

                cv2.putText(image, 'DOUBLE BICEP CURLS', (390,18), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (137, 207, 240), 1, cv2.LINE_AA)

                if counter >=1 and counter <2:
                            cv2.putText(image, '1 Set completed ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

                if counter >=2 and counter <3:
                            cv2.putText(image, '2 Set completed ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

                if counter ==3 :
                            cv2.putText(image, '3 Set completed ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

                if counter >3 :
                            cv2.rectangle(image, (40,200), (600,72), (255,255,255), -1)


                            cv2.putText(image, 'Please Stop Workout!!!', (60,150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 1, cv2.LINE_AA)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        )
            


                #setting up curl counter box
                cv2.rectangle(image, (0,0), (360,72), (245,117,16), -1)

                #sending values to curl counter box
                cv2.putText(image, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), 
                            (10,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                #printing hand stage while exercising

                cv2.putText(image, 'STAGE', (165,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, 
                            (165,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                cv2.putText(image, 'HIGH KNEES', (540,18), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

                if counter >=1 and counter <2:
                            cv2.putText(image, '1 Set completed ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter >=2 and counter <3:
                            cv2.putText(image, '2 Set completed ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter ==3 :
                            cv2.putText(image, '3 Set completed ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter >3 :

                            cv2.rectangle(image, (40,200), (600,72), (255,255,255), -1)


                            cv2.putText(image, 'Please Stop Workout!!!', (60,150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 1, cv2.LINE_AA)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        )     
        
                cv2.imshow('Webcam Video - Start Excercise', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break        
        

    cap.release()
    cv2.destroyAllWindows() 
    return render(request, "camera.html")


def camera2_vid(request): #Excerise: 1. Bicep_curl 
    mp_drawing=mp.solutions.drawing_utils
    mp_pose=mp.solutions.pose
    #mod = Videos.objects.values_list('video', flat=True).distinct()[0]
    mod = Videos.objects.latest('video')

    m = str(mod)
    # mm = ("/home/atif/Desktop/django website gym/media/" + m)
    # mm = ("/home/atif/Desktop/django website gym/media/" + last_field_value) 
    v2 = mm
    print(v2)
    #'Excerise: 1. Bicep_curl 2. Shoulder_press 3. Highknees'
    if 5 ==5:    
        choice = 2
        def calculate_angle(a,b,c):
            a = np.array(a)  #first angle
            b = np.array(b)  #second angle
            c = np.array(c)  #third angle
            
            radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle>180.0:
                angle = 360-angle
            return angle

        #Curl Counter
        cap = cv2.VideoCapture(v2)
        counter = 0
        stage = None
        error=0
        ## Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                #Selection of Excercise
                if(choice==1):
                    try:
                        landmarks = results.pose_landmarks.landmark

                        # Get coordinates
                        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                        # Calculate angle
                        angle_1 =calculate_angle(left_shoulder, left_elbow, left_wrist)
                        angle_2=calculate_angle(right_shoulder,right_elbow,right_wrist)

                        # Visualize angle
                        cv2.putText(image, str(angle_1), 
                                    tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                            )

                        cv2.putText(image, str(angle_2), 
                                    tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                            )

                        #CURL COUNTER LOGIC
                        if angle_1 > 160 and angle_2 >160 :
                            stage = "Down"
                        if angle_1 < 30 and angle_2 < 30 and stage == 'Down':
                            stage = "up"
                            counter +=1
                            print(counter)

                    except:
                        pass

                    #setting up curl counter box
                    cv2.rectangle(image, (0,0), (360,72), (245,117,16), -1)

                    #sending values to curl counter box
                    cv2.putText(image, 'REPS', (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(counter), 
                                (10,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

                    #printing hand stage while exercising

                    cv2.putText(image, 'STAGE', (165,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, stage, 
                                (165,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

                    cv2.putText(image, 'DOUBLE BICEP CURLS', (390,18), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (137, 207, 240), 1, cv2.LINE_AA)

                    if counter >=1 and counter <2:
                                cv2.putText(image, '1 Set completed ', (390,90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

                    if counter >=2 and counter <3:
                                cv2.putText(image, '2 Set completed ', (390,90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

                    if counter ==3 :
                                cv2.putText(image, '3 Set completed ', (390,90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

                    if counter >3 :
                                cv2.rectangle(image, (40,200), (600,72), (255,255,255), -1)


                                cv2.putText(image, 'Please Stop Workout!!!', (60,150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 1, cv2.LINE_AA)

                    # Render detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                            )
                elif(choice==2):
                        try:
                            landmarks = results.pose_landmarks.landmark

                            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                            # Calculate angle
                            angle_1 = calculate_angle(left_shoulder, left_elbow, left_wrist)
                            angle_2=calculate_angle(right_shoulder,right_elbow,right_wrist)

                            # Visualize angle
                            cv2.putText(image, str(angle_1), 
                                        tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                                )

                            cv2.putText(image, str(angle_2), 
                                        tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                                )

                            #CURL COUNTER LOGIC
                            if angle_1 > 110 and angle_2 >110 :
                                stage = "Up"
                            if angle_1 < 90 and angle_2 < 90 and stage == 'Up':
                                stage = "Down"
                                counter +=1
                                print(counter)


                        except:
                            pass

                        #setting up curl counter box
                        cv2.rectangle(image, (0,0), (360,72), (245,117,16), -1)

                        #sending values to curl counter box
                        cv2.putText(image, 'REPS', (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(counter), 
                                    (10,60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                        #printing hand stage while exercising

                        cv2.putText(image, 'STAGE', (165,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        cv2.putText(image, stage, 
                                    (165,60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                        cv2.putText(image, 'SHOULDER_PRESS', (390,18), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                        if counter >=1 and counter <2:
                                    cv2.putText(image, '1 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

                        if counter >=2 and counter <3:
                                    cv2.putText(image, '2 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

                        if counter ==3 :
                                    cv2.putText(image, '3 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

                        if counter >3 :

                                    cv2.rectangle(image, (40,200), (600,72), (255,255,255), -1)


                                    cv2.putText(image, 'Please Stop Workout!!!', (60,150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 1, cv2.LINE_AA)

                        # Render detections
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                                )            
                
                elif(choice==3):
                        
                        # Extract landmarks
                        try:
                            landmarks = results.pose_landmarks.landmark

                            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                            # Calculate angle
                            angle = calculate_angle(shoulder, hip, knee)

                            # Visualize angle
                            cv2.putText(image, str(angle), 
                                        tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                                )

                            #CURL COUNTER LOGIC
                            if angle > 160:
                                stage = "Down"
                            if angle < 80 and stage == 'Down':
                                stage = "Up"
                                counter +=1
                                print(counter)


                        except:
                            pass

                        #setting up curl counter box
                        cv2.rectangle(image, (0,0), (360,72), (245,117,16), -1)

                        #sending values to curl counter box
                        cv2.putText(image, 'REPS', (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(counter), 
                                    (10,60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                        #printing hand stage while exercising

                        cv2.putText(image, 'STAGE', (165,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        cv2.putText(image, stage, 
                                    (165,60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                        cv2.putText(image, 'HIGH KNEES', (540,18), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

                        if counter >=1 and counter <2:
                                    cv2.putText(image, '1 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                        if counter >=2 and counter <3:
                                    cv2.putText(image, '2 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                        if counter ==3 :
                                    cv2.putText(image, '3 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                        if counter >3 :

                                    cv2.rectangle(image, (40,200), (600,72), (255,255,255), -1)


                                    cv2.putText(image, 'Please Stop Workout!!!', (60,150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 1, cv2.LINE_AA)

                        # Render detections
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                                )     
                else:
                    print("Invalid choice!!!")
                    break
                
                cv2.imshow('Webcam Video - Start Excercise', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                
            

            cap.release()
            cv2.destroyAllWindows()
    return render(request, 'camera.html')       

def camera4_vid(request):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    #mod = Videos.objects.values_list('video', flat=True).distinct()[0]
    mod = Videos.objects.latest('video')
    m = str(mod)
    # mm = ("/home/atif/Desktop/django website gym/media/" + m)
    mm = ("/home/atif/Desktop/django website gym/media/" + last_field_value) 
    print(mm)
    v4 = mm
    
    def calculate_angle(a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle = 360-angle
            
        return angle


    def rescale_frame(frame, percent=50):
        width = int(frame.shape[1] * percent/ 100)
        height = int(frame.shape[0] * percent/ 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


    # Youtube video
    angle_min = []
    angle_min_hip = []
    cap = cv2.VideoCapture("demo.mp4")
    # Curl counter variables
    counter = 0 
    min_ang = 0
    max_ang = 0
    min_ang_hip = 0
    max_ang_hip = 0
    stage = None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (640, 480)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('demo.mp4', fourcc, 24, size)


    cap = cv2.VideoCapture(v4)
    # Curl counter variables
    counter = 0 
    stage = None

    """width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('your_video.mp4', fourcc, 10.0, size, True)"""

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                
                # Get coordinates
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
            
                
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                
                angle_knee = calculate_angle(hip, knee, ankle) #Knee joint angle
                
                angle_hip = calculate_angle(shoulder, hip, knee)
                hip_angle = 180-angle_hip
                knee_angle = 180-angle_knee
                
                # Visualize angle
                """cv2.putText(image, str(angle), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )"""
                        
                    
                cv2.putText(image, str(angle_knee), 
                            tuple(np.multiply(knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (79, 121, 66), 2, cv2.LINE_AA
                                    )
                
                """cv2.putText(image, str(angle_hip), 
                            tuple(np.multiply(hip, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )"""
                
                # Curl counter logic
                if angle_knee > 169:
                    stage = "UP"
                if angle_knee <= 90 and stage =='UP':
                    stage="DOWN"
                    counter +=1
                    print(counter)
            except:
                pass
            
            # Render squat counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (65,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (60,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            out.write(image)
            cv2.imshow('Squats Excercise', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        #out.release()
        cv2.destroyAllWindows()
            
        
    return render(request, "camera.html")

def camera3_vid(request): #Excerise: 1. Bicep_curl 
    mp_drawing=mp.solutions.drawing_utils
    mp_pose=mp.solutions.pose
    #mod = Videos.objects.values_list('video', flat=True).distinct()[0]
    mod = Videos.objects.latest('video')
    m = str(mod)
    # mm = ("/home/atif/Desktop/django website gym/media/" + m)
    # mm = ("/home/atif/Desktop/django website gym/media/" + last_field_value) 
    v3 = mm
    
    #'Excerise: 1. Bicep_curl 2. Shoulder_press 3. Highknees'
    if 5 ==5:    
        choice = 3
        def calculate_angle(a,b,c):
            a = np.array(a)  #first angle
            b = np.array(b)  #second angle
            c = np.array(c)  #third angle
            
            radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle>180.0:
                angle = 360-angle
            return angle

        #Curl Counter
        cap = cv2.VideoCapture(v3)
        counter = 0
        stage = None
        error=0
        ## Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                #Selection of Excercise
                if(choice==1):
                    try:
                        landmarks = results.pose_landmarks.landmark

                        # Get coordinates
                        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                        # Calculate angle
                        angle_1 =calculate_angle(left_shoulder, left_elbow, left_wrist)
                        angle_2=calculate_angle(right_shoulder,right_elbow,right_wrist)

                        # Visualize angle
                        cv2.putText(image, str(angle_1), 
                                    tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                            )

                        cv2.putText(image, str(angle_2), 
                                    tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                            )

                        #CURL COUNTER LOGIC
                        if angle_1 > 160 and angle_2 >160 :
                            stage = "Down"
                        if angle_1 < 30 and angle_2 < 30 and stage == 'Down':
                            stage = "up"
                            counter +=1
                            print(counter)

                    except:
                        pass

                    #setting up curl counter box
                    cv2.rectangle(image, (0,0), (360,72), (245,117,16), -1)

                    #sending values to curl counter box
                    cv2.putText(image, 'REPS', (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(counter), 
                                (10,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

                    #printing hand stage while exercising

                    cv2.putText(image, 'STAGE', (165,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, stage, 
                                (165,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

                    cv2.putText(image, 'DOUBLE BICEP CURLS', (390,18), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (137, 207, 240), 1, cv2.LINE_AA)

                    if counter >=1 and counter <2:
                                cv2.putText(image, '1 Set completed ', (390,90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

                    if counter >=2 and counter <3:
                                cv2.putText(image, '2 Set completed ', (390,90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

                    if counter ==3 :
                                cv2.putText(image, '3 Set completed ', (390,90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

                    if counter >3 :
                                cv2.rectangle(image, (40,200), (600,72), (255,255,255), -1)


                                cv2.putText(image, 'Please Stop Workout!!!', (60,150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 1, cv2.LINE_AA)

                    # Render detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                            )
                elif(choice==2):
                        try:
                            landmarks = results.pose_landmarks.landmark

                            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                            # Calculate angle
                            angle_1 = calculate_angle(left_shoulder, left_elbow, left_wrist)
                            angle_2=calculate_angle(right_shoulder,right_elbow,right_wrist)

                            # Visualize angle
                            cv2.putText(image, str(angle_1), 
                                        tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                                )

                            cv2.putText(image, str(angle_2), 
                                        tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                                )

                            #CURL COUNTER LOGIC
                            if angle_1 > 110 and angle_2 >110 :
                                stage = "Up"
                            if angle_1 < 90 and angle_2 < 90 and stage == 'Up':
                                stage = "Down"
                                counter +=1
                                print(counter)


                        except:
                            pass

                        #setting up curl counter box
                        cv2.rectangle(image, (0,0), (360,72), (245,117,16), -1)

                        #sending values to curl counter box
                        cv2.putText(image, 'REPS', (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(counter), 
                                    (10,60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                        #printing hand stage while exercising

                        cv2.putText(image, 'STAGE', (165,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        cv2.putText(image, stage, 
                                    (165,60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                        cv2.putText(image, 'SHOULDER_PRESS', (390,18), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                        if counter >=1 and counter <2:
                                    cv2.putText(image, '1 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

                        if counter >=2 and counter <3:
                                    cv2.putText(image, '2 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

                        if counter ==3 :
                                    cv2.putText(image, '3 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

                        if counter >3 :

                                    cv2.rectangle(image, (40,200), (600,72), (255,255,255), -1)


                                    cv2.putText(image, 'Please Stop Workout!!!', (60,150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 1, cv2.LINE_AA)

                        # Render detections
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                                )            
                
                elif(choice==3):
                        
                        # Extract landmarks
                        try:
                            landmarks = results.pose_landmarks.landmark

                            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                            # Calculate angle
                            angle = calculate_angle(shoulder, hip, knee)

                            # Visualize angle
                            cv2.putText(image, str(angle), 
                                        tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                                )

                            #CURL COUNTER LOGIC
                            if angle > 160:
                                stage = "Down"
                            if angle < 80 and stage == 'Down':
                                stage = "Up"
                                counter +=1
                                print(counter)


                        except:
                            pass

                        #setting up curl counter box
                        cv2.rectangle(image, (0,0), (360,72), (245,117,16), -1)

                        #sending values to curl counter box
                        cv2.putText(image, 'REPS', (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(counter), 
                                    (10,60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                        #printing hand stage while exercising

                        cv2.putText(image, 'STAGE', (165,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        cv2.putText(image, stage, 
                                    (165,60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                        cv2.putText(image, 'HIGH KNEES', (540,18), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

                        if counter >=1 and counter <2:
                                    cv2.putText(image, '1 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                        if counter >=2 and counter <3:
                                    cv2.putText(image, '2 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                        if counter ==3 :
                                    cv2.putText(image, '3 Set completed ', (390,90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                        if counter >3 :

                                    cv2.rectangle(image, (40,200), (600,72), (255,255,255), -1)


                                    cv2.putText(image, 'Please Stop Workout!!!', (60,150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 1, cv2.LINE_AA)

                        # Render detections
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                                )     
                else:
                    print("Invalid choice!!!")
                    break
                
                cv2.imshow('Webcam Video - Start Excercise', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                
            

            cap.release()
            cv2.destroyAllWindows()
    return render(request, 'camera.html')  


########################################################################################
     
def fl_camera(request):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    # Variables for tracking arm positions
    landmarks_formed = False
    right_hand_raised = False
    left_hand_raised = False

    cap = cv2.VideoCapture(0)
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
     while cap.isOpened():
        success, image = cap.read()
        font = cv2.FONT_HERSHEY_SIMPLEX
        stage = None
        org = (50, 50)
        fontScale = 1
        color = (0, 0, 0)
        thickness = 2
        if not success:
            print("Ignoring empty camera frame.")
            continue
        # To improve performance, optionally mark the image as not writeable to
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)


        if(results.pose_landmarks):
        
            # Indicate that a user has been detected.
            if(not landmarks_formed):
                landmarks_formed = True
                print("Person detected. Begin arm movements.")

        # Store importnat landmarks
        right_shoulder = results.pose_landmarks.landmark[12]
        right_wrist = results.pose_landmarks.landmark[16]
        left_shoulder = results.pose_landmarks.landmark[11]
        left_wrist = results.pose_landmarks.landmark[15]

        # If the right wrist raises above the right shoulder, it is raised.
        if(right_wrist.y < right_shoulder.y and not right_hand_raised):
            stage = "Right Hand"
            print("Right hand raised")
            right_hand_raised = True

        # If the right wrist falls below the right shoulder, it is lowered.
        if(right_wrist.y > right_shoulder.y and right_hand_raised):
            stage = "Right Hand down"
            print("Right hand lowered")
            right_hand_raised = False
        
        # If the left wrist raises above the left shoulder, it is raised.
        if(left_wrist.y < left_shoulder.y and not left_hand_raised):
            stage = "Left hand raised"
            print("Left hand raised")
            left_hand_raised = True

        # If the left wrist falls below the left shoulder, it is lowered.
        if(left_wrist.y > left_shoulder.y and left_hand_raised):
            stage = "Left hand lowered"
            print("Left hand lowered")
            left_hand_raised = False
        
        # If landmarks cannot be formed (no person detected)...
        elif(landmarks_formed):
            landmarks_formed = False
            stage = "Start Hand Excercise"
            # print("No person detected. Please re-enter the frame.")


                                
        cv2.putText(image, 'STAGE', (165,12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                (165,60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
            


        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # Flip the image horizontally for a selfie-view display.

        
        cv2.imshow('Hand Exercises & Position Detector', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    return render(request, 'camera.html') 




def camera5_vid(request):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    # Variables for tracking arm positions
    landmarks_formed = False
    right_hand_raised = False
    left_hand_raised = False
    #mod = Videos.objects.latest('video')   # to pick last video from the table.
    # mm = ("/home/atif/Desktop/django website gym/media/" + last_field_value) 
    
    cap = cv2.VideoCapture(mm)
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
     while cap.isOpened():
        success, image = cap.read()
        font = cv2.FONT_HERSHEY_SIMPLEX
        stage = None
        org = (50, 50)
        fontScale = 1
        color = (0, 0, 0)
        thickness = 2
        if not success:
            print("Ignoring empty camera frame.")
            continue
        # To improve performance, optionally mark the image as not writeable to
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        
        
        right_wrist = None
        if(results.pose_landmarks):
        
            # Indicate that a user has been detected.
            if(not landmarks_formed):
                landmarks_formed = True
                print("Person detected. Begin arm movements.")

            # Store importnat landmarks
            right_shoulder = results.pose_landmarks.landmark[12]
            right_wrist = results.pose_landmarks.landmark[16]
            left_shoulder = results.pose_landmarks.landmark[11]
            left_wrist = results.pose_landmarks.landmark[15]

        # If the right wrist raises above the right shoulder, it is raised.
        if(right_wrist.y < right_shoulder.y and not right_hand_raised):
            stage = "Right Hand"
            print("Right hand raised")
            right_hand_raised = True

        # If the right wrist falls below the right shoulder, it is lowered.
        if(right_wrist.y > right_shoulder.y and right_hand_raised):
            stage = "Right Hand down"
            print("Right hand lowered")
            right_hand_raised = False
        
        # If the left wrist raises above the left shoulder, it is raised.
        if(left_wrist.y < left_shoulder.y and not left_hand_raised):
            stage = "Left hand raised"
            print("Left hand raised")
            left_hand_raised = True

        # If the left wrist falls below the left shoulder, it is lowered.
        if(left_wrist.y > left_shoulder.y and left_hand_raised):
            stage = "Left hand lowered"
            print("Left hand lowered")
            left_hand_raised = False
        
        # If landmarks cannot be formed (no person detected)...
        elif(landmarks_formed):
            landmarks_formed = False
            stage = "Start Hand Excercise"
            print("No person detected. Please re-enter the frame.")


                                
        cv2.putText(image, 'STAGE', (165,12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                (165,60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
            


        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # Flip the image horizontally for a selfie-view display.

        
        cv2.imshow('Hand Exercises & Position Detector', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    return render(request, 'camera.html') 