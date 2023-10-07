import streamlit as st
from PIL import Image
import numpy as np
import cv2, random

import os, math, sys
import torch
from torchvision import transforms
import torch.nn as nn
from torchvision.models import efficientnet_b0 


class Application:
    def __init__(self, video_capture):
        self.video_capture = video_capture
        self.flag = "im_upload" 

        prototxt_path = "src\\deploy.prototxt.txt"
        model_path = "src\\res10_300x300_ssd_iter_140000_fp16.caffemodel"
        self.face_model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

        # face_cascade=cv2.CascadeClassifier('src\\cascades\\haarcascade_frontalface_default.xml')
        self.eye_cascade= cv2.CascadeClassifier('src\\cascades\\haarcascade_eye.xml')

        # assert not face_cascade.empty() 
        assert not self.face_model.empty() 
        assert not self.eye_cascade.empty() 

        #load smiling model
        self.preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        #score map using label inverse transform from training code
        self.smiling_score_map= {
            0 : 1, #huge smile
            1 : 0.5, #mildly smiling
            2: 0.7, #moderatly smiling
            3: 0.2 #not smiling
        } 

        self.smiling_class_map = {
            0 : "huge smile",
            1 : "mildly smiling",
            2:  "moderatly smiling",
            3: "not smiling",
        } 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.smiling_model = efficientnet_b0(weights=False)
        num_features = self.smiling_model._modules["classifier"][1].in_features
        self.smiling_model._modules["classifier"][1] = nn.Sequential(
                nn.Linear(num_features, 64),
                nn.ReLU(),
                nn.Linear(64, 4),
                )

        checkpoint = torch.load("src\\smile_model.pth")
        self.smiling_model.load_state_dict(checkpoint)
        self.smiling_model.to(self.device)

        self.face_count, self.eye_count, self.smiling_scores = 0,0, []


    def get_printability_score(self):
        print(self.face_count, self.eye_count, self.smiling_scores)
        avg_smile_score = 0 
        if len(self.smiling_scores)!= 0:
            avg_smile_score =  sum(self.smiling_scores) / len(self.smiling_scores)
        else:
            avg_smile_score = 0

        open_eye_ratio = None
        if self.face_count == 0:
            open_eye_ratio = 0
        else:
            open_eye_ratio = 0.5 *(self.eye_count / self.face_count ) #scaled to 1

        print(f"Avg smile score : {avg_smile_score}, Open eye ratio: { open_eye_ratio}")
        
        score = round(10 * ( (0.8* avg_smile_score ) + ( 0.2 * open_eye_ratio) ), 1)
        is_smiling = True if (avg_smile_score >= 0.4) else False 
        eyes_open = True if (open_eye_ratio >= 0.8) else False

        return  f"Score: {score}, Smiling: {is_smiling}, Eyes Open: {eyes_open}"


    def get_smiling_score(self, frame):
        frame = Image.fromarray(frame)
        input_tensor = self.preprocess(frame)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.smiling_model(input_batch)
            _, predicted = torch.max(output.data, 1)
            print(output, predicted)
            
            print(self.smiling_class_map[predicted[0].item()])
            self.smiling_scores.append(self.smiling_score_map[predicted[0].item()])


    def get_cascades(self, frame):
        #get image cascades and process variables for printability rating
        self.face_count, self.eye_count, self.smiling_scores = 0,0, [] #reset
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5 )

        h, w = gray.shape
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_model.setInput(blob)
        output = np.squeeze(self.face_model.forward())

        faces = []
        for i in range(output.shape[0]):
            confidence = output[i, 2]
            if confidence > 0.5:
                # get the surrounding box cordinates and upscale them to original image
                box = output[i, 3:7] * np.array([w, h, w, h])
                faces.append(box.astype(int))


        for (x, y, w, h) in faces:
            self.get_smiling_score(frame[y:y+h, x:x+w])

            # Draw a rectangle around the face
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Crop the region of interest for eyes within the detected face
            roi_gray = gray[y:y + h, x:x + w]

            # Detect eyes in the region of interest
            eyes = self.eye_cascade.detectMultiScale(roi_gray, minNeighbors = 6, minSize=(30,30))

            if len(eyes) > 2:
                self.eye_count += 2
            else:
                self.eye_count += len(eyes)

            for (ex, ey, ew, eh) in eyes:
                # Draw rectangles around the left and right eyes
                frame = cv2.rectangle(frame, (ex+x, ey+y), (ex + ew + x, ey + eh + y), (0, 0, 255), 3)

        self.face_count = len(faces)
        

        return frame


    def process_image(self, image):
        #get the cascades and apply it
        image = np.array(image)
        image = self.get_cascades(image)
        return image


    def im_upload(self):
        st.header("Image Upload")
        uploaded_images = st.file_uploader("Upload one or more images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

        if uploaded_images:
            st.write("Uploaded Images:")

            for image in uploaded_images:

                image = Image.open(image).convert("RGB")
                image = self.process_image(image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                st.write(self.get_printability_score()) 



    

    def live_video(self):
        st.header("Live Video")
        
        img = st.image([])
        container = st.empty()

        while self.interface_option == "Live Video":
            ret, frame = self.video_capture.read()
            if not ret:
                print("Could not read frame from cv capture.")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
            img.image(self.process_image(frame))
            container.empty()
            container.text(self.get_printability_score())

    def window(self):
        st.title("Image Printability Rating")

        self.interface_option = st.radio("Select Interface", ["Image Upload",  "Live Video"])

        if self.interface_option == "Image Upload":
            self.im_upload()
        else:
            self.live_video()



if __name__ == "__main__":
    
    video_capture = cv2.VideoCapture(0)
    try:
        app = Application(video_capture)
        app.window()
    except KeyboardInterrupt as e:
        video_capture.release()

