import threading
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
from ultralytics import YOLO
import streamlit as st
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image
import supervision as sv

#####################################################################################################################
# DEPLOY MODEL!!!!!!

#Load model
model = YOLO("best.pt")
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

#function to detect
def detect_objects(image):
    #image = Image.fromarray(image).convert('RGB')
    result = model.predict(image)[0]
    detections = sv.Detections.from_ultralytics(result)

    labels = [
        model.model.names[class_id]
        for class_id
        in detections.class_id
    ]

    annotated_image = bounding_box_annotator.annotate(
        scene =image,
        detections = detections
    )

    annotated_image = label_annotator.annotate(
        scene = annotated_image,
        detections = detections,
        labels = labels
    )

    return av.VideoFrame.from_ndarray(annotated_image, format = "bgr24");
#####################################################################################################################

#####################################################################################################################
#OUTLINE OF WEBSITE !!!!!!!!

#INDIVIDUAL TABS
def infoTAB():
    st.write("Learn More")

def demoTAB():
    st.title("Trashy classifier")
    option = st.radio("Choose Input Method:", ('Upload Image', 'Use Camera'))
    if option == 'Upload Image':
        uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            results = detect_objects(image)
            st.image(results)  
    else:
        #st.title("Webcam Live Feed")
        #run = st.button('Run')
        #end = st.button('End Feed')
        #FRAME_WINDOW = st.image([])
        #camera = cv2.VideoCapture(0)
        webrtc_streamer(key="example")
            #_, frame = camera.read()
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #results = detect_objects(frame)
            #FRAME_WINDOW.image(results)
        # if end:
        #     st.write('Webcam Stopped')
        #     camera.release()
        #     cv2.destroyAllWindows()
    
        


#####################################################################################################################
#  RUN STREAMLIT !!!!!!!!!
def main():
    tab_select = st.sidebar.radio("Welcome! Where would you like to explore?", ("Know More About Our Model!", "Try Out Our Model!"))
    if tab_select == "Know More About Our Model!":
        infoTAB()
    elif tab_select == "Try Out Our Model!":
        demoTAB()
if __name__ == '__main__':
    main()
#####################################################################################################################



