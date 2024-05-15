import tkinter as tk
from tkinter import filedialog
from tkinter import *
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from PIL import Image,ImageTk
import numpy as np
import cv2

def FacialExpressionModel(json_file,weights_file):
    with open(json_file,"r") as file:
        loaded_model_json=file.read()
        model=model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer='adam',loss='categorical_crossenropy',metrics=['accuracy'])
    return model

top=tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

#label1=Label(top,background='#CDCDCD',font=('arial',15,'bold'))
label2=Label(top,background='#CDCDCD',font=('arial',15,'bold'))

#label1.place(x=500, y=50)
sign_image=Label(top)
#sign_image.place(x=100, y=200)

facec=cv2.CascadeClassifier('haarcascad_frontalface_default.xml')
model=FacialExpressionModel('model_a.json','model_weights.h5')

EMOTIONS_LIST=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprice']
def Detect(file_path):
    global Label_packed
    
    image=cv2.imread(file_path)
    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=facec.detectMultiScale(gray_image,1.3,5)
    try:
        for (x,y,w,h) in faces:
            fc=gray_image[y:y+h,x:x+w]
            roi=cv2.resize(fc,(48,48))
            pred=EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis,:,:,np.newaxis]))]
            print('Predicted emotion is' + pred)
            label2.configure(foreground="#364156",text=pred)
    except:
        label2.configure(foreground="#364156",text="unable to detect")
        

def show_detect_button(file_path):
    detect_b=Button(top,text="Detect Emotion", command=lambda:Detect(file_path),padx=10,pady=5)
    detect_b.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
    detect_b.place(relx=0.79,rely=0.46)
    
def upload_Image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.5),(top.winfo_height()/2.5)))
        #uploaded.thumbnail((150, 200))  # This will resize the image to a maximum width of 200 and a maximum height of 150

        im=ImageTk.PhotoImage(uploaded)
        
        sign_image.configure(image=im)
        sign_image.image=im
        label2.configure(text='aaa')
        show_detect_button(file_path)
    except:
        pass
upload=Button(top,text="Upload Image",command=upload_Image,padx=10,pady=50)
upload.configure(background='#364156',foreground='white',font=('arial',16,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom',expand='True')
#label1.pack(side='bottom',expand='True')
heading=Label(top,text='Emotion Detector',pady=10,font=('arial',25,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()
    


