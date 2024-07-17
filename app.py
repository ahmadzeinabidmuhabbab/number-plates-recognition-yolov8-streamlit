import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
from post_processing import post_processing
# import cv2

def predict(image, conf, iou):
    model = YOLO(f"best.pt")
    results = model.predict(source=image, conf=conf, iou=iou)
    names = model.names
    labels = []

    for r in results:
        # Get bounding boxes and class indices
        boxes = r.boxes.xyxy
        cls = r.boxes.cls

        # Sort bounding boxes and class indices by x1 coordinate
        sorted_indices = boxes[:, 0].argsort()
        sorted_boxes = boxes[sorted_indices]
        sorted_cls = cls[sorted_indices]

        # Get class names
        file_labels = [names[int(c)] for c in sorted_cls]

        # Append labels to list
        labels.append(file_labels)
    
    label = "".join(labels[0])

    return label

conf = st.slider('Select a confidence value between 0 and 1', 0.0, 1.0, 0.3, 0.01)
iou = st.slider('Select a IoU value between 0 and 1', 0.0, 1.0, 0.7, 0.01)

img_file_buffer = st.file_uploader('Upload a PNG image', type=['png', 'jpg', 'jpeg'])
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    image_resize = image.resize((640,640))
    predict_yolo = predict(image_resize, conf, iou)
    df = pd.DataFrame({'Labels':[predict_yolo]})
    df_post = post_processing(df)
    df_post_final = df_post[['Labels','mod_labels']]
    df_post_final.columns = ['Before Post-Processing','After Post-Processing']
    # st.write(predict_yolo)
    st.dataframe(df_post_final, use_container_width=True)