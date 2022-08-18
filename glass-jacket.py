import cv2,sys,os,time,dlib
import numpy as np
import dlib
from PIL import Image, ImageOps
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import statistics
import pyttsx3
import pywhatkit
import speech_recognition as sr
import datetime
import webbrowser
import os

#**Voice Assistant**
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)


def speak(audio):
    engine.say(audio)
    engine.runAndWait()
    
def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour>= 0 and hour<12:
        speak("hallo,Guten Morgen !")
  
    elif hour>= 12 and hour<18:
        speak("hallo, Guten Tag!")  
  
    else:
        speak("hallo, guten Abend!") 
  
    speak("Ich bin dein Assistent")
   
         
    
    
def takeCommand():
     
    r = sr.Recognizer()
     
    with sr.Microphone() as source:
         
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)
  
    try:
        print("Recognizing...")   
        query = r.recognize_google(audio, language ='de')
        print(f"User said: {query}\n")
  
    except Exception as e:
        print(e)   
        print("Unable to Recognize your voice.") 
        return "None"
     
    return query 



def prep_image(pred_image):
    
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    image = Image.fromarray(pred_image)
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array


    
    return data


if __name__ == '__main__':

    
    start = time.time()
         
    query = takeCommand().lower()
         
        # All the commands said by user will be
        # stored here in 'query' and will be
        # converted to lower case for easily
        # recognition of command


    #**Voice Assistant**

    FACE_DOWNSAMPLE_RATIO = 1
    RESIZE_HEIGHT = 360

    predictions2Label = {0:"No Glasses", 1:"With Glasses"}
    jacket_class_names = ['With Jacket','No Jacket']

  # Load face detection, predictor and jacket models.
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('C:/safety_new_project/FINAL/shape_predictor_68_face_landmarks.dat')
    jacket_model = load_model('C:/safety_new_project/FINAL/keras_model.h5',compile=False)
  #cap = cv2.VideoCapture(0)
    cap =  cv2.VideoCapture('C:/safety_new_project/FINAL/jacket_video_Trim.mp4')

  # Check if webcam opens
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    while(1):
        clear = lambda: os.system('cls')
         
        # This Function will clean any
        # command before execution of this python file
        clear()
        wishMe()    
        safe=0
        unsafe=0
        _ = 0
        while(1):
      
            try:
              
              t = time.time()
              # Read frame
              ret, frame = cap.read()
              height, width = frame.shape[:2]
              
              img= frame
              
              rect = detector(img)[0]
              sp = predictor(img, rect)
              landmarks = np.array([[p.x, p.y] for p in sp.parts()])
              #trying get box to do a crop around the detected nose
              nose_bridge_x = []
              nose_bridge_y = []
              nose_bridge_x = []
              nose_bridge_y = []
              for i in [28,29,30,31,33,34,35]:
                 nose_bridge_x.append(landmarks[i][0])
                 nose_bridge_y.append(landmarks[i][1])
            ### x_min and x_max
              x_min = min(nose_bridge_x)
              x_max = max(nose_bridge_x)
              y_min = landmarks[20][1]
              y_max = landmarks[31][1]
              img2 = Image.fromarray(frame)
              img2 = img2.crop((x_min,y_min,x_max,y_max))
              plt.imshow(img2)
              #applying gaussianblur and canny filter to get some highlighted lines around the nose
              img_blur = cv2.GaussianBlur(np.array(img2),(3,3), sigmaX=0, sigmaY=0)
              edges = cv2.Canny(image =img_blur, threshold1=100, threshold2=200)
              plt.imshow(edges, cmap =plt.get_cmap('gray'))
              edges_center = edges.T[(int(len(edges.T)/2))]
              
              glass_pred=''
              if 255 in edges_center:
                 glass_class = predictions2Label[1]
              else:
                 glass_class = predictions2Label[0]
                 
              #print predictions
              dt = prep_image(frame)
              
              ll = list(jacket_model.predict(dt)[0])
              dd = dict(zip(jacket_class_names,ll))
              jacket_class = max(zip(dd.values(), dd.keys()))[1]
              print(jacket_class)
                    
              #frameClone = np.copy(frame)
              #cv2.putText(frameClone, "Prediction = {}".format(predictions2Label[int(predictions[0])]), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
              
              
              frameClone = np.copy(frame)
            #puttext glass class and jacket class that has been predicted
              cv2.putText(frameClone, glass_class, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,0), 2)
              cv2.putText(frameClone, jacket_class, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,0), 2)
            
            
              print(glass_class)
            
            ##puttext safe or not
           
              if(jacket_class==jacket_class_names[0]) & (glass_class==predictions2Label[1]):
                  #cv2.putText(frameClone, 'Safe', (10, frameClone.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0), 2)
                  safe=safe+1
                  print('**'+str(safe)+'**')
              else:
                  #cv2.putText(frameClone, 'Not Safe', (10, frameClone.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,0), 2)
                  unsafe=unsafe+1
                  print('##'+str(unsafe)+'##')
              
              cv2.imshow("Original Frame", frameClone)
              # cv2.imshow("Eye", cropped)
              #if cv2.waitKey(1) & 0xFF == 27:
              print("Total time : {}".format(time.time() - t))
           
              _+= time.time() - t
              
            
              print("timer : {}".format(_))
              if _>25:
                break

         
            except Exception as e:
                #incase nose is not detected or some other error
              frameClone = np.copy(frame)
              #cv2.putText(frameClone, "Face Not detected properly", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
              cv2.imshow("Original Frame", frameClone)
        #      cv2.imshow("Eye", cropped)
              if _>25:
                break
              print(e)
          
        if unsafe>5:
         speak('unsafe')

        elif safe>5 :
         speak('safe')  
         
        else :
         speak('unknown')  


      
      
    
    cv2.destroyAllWindows()