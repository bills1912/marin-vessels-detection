import time
from turtle import width
# import torch
import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static 

st.set_page_config(
        page_title="Ship Detection using YOLOv5 Medium Model",
        page_icon=":ship:",
        layout="wide"
    )

st.write("# Welcome to Ship Detection Application! :satellite:")
st.markdown(
        """
        This application is build based on YOLOv5 with extral large model. User just
        upload an image, and press the 'Predict' button to make a prediction base on
        a training model before.

        ### For more information, please visit:

        - Check out [my github](https://github.com/bills1912)
        - Jump into YOLOv5 [documentation](https://docs.ultralytics.com/)

    """
    )

st.write("## Ship Imagery Prediction")

ais = pd.read_csv("https://raw.githubusercontent.com/bills1912/marin-vessels-detection/main/data/MarineTraffic_VesselExport_2022-09-08.csv")
ais_jakarta = ais[ais['Destination Port'] == 'JAKARTA']
ais_list = ais_jakarta.values.tolist()
f = folium.Figure(width=1000, height=500)
jakarta_vessels = folium.Map(location=[-5.626954250925966, 106.70735731868719], zoom_start=8).add_to(f)
ais_data = folium.FeatureGroup(name="marine_vessels")
mCluster = MarkerCluster(name="Marine Vessels")
for i in ais_list:
  html = f"<h3>{i[1]}</h3> Vessel Type: {i[7]} </br> Destination Port: {i[2]} </br> Reported Destination: {i[4]} </br> Current Port: {i[5]}\
          </br> Latitude: {i[9]} </br> Longitude: {i[10]}"
  iframe = folium.IFrame(html)
  popup = folium.Popup(iframe, min_width=250, max_width=300)
  ais_data.add_child(mCluster.add_child(folium.Marker(location=[i[9], i[10]], popup=popup, icon=folium.Icon(color="black", icon="ship", prefix="fa"))))
jakarta_vessels.add_child(ais_data)
folium_static(jakarta_vessels, width=1000, height=700)


st.write("### Model evaluation:")
eval_col1, eval_col2, eval_col3, eval_col4 = st.columns(spec=4)
eval_col1.metric("Precision", "94.064%")
eval_col2.metric("Recall", "96.053%")
eval_col3.metric("mAP 0.5", "98.56%")
eval_col4.metric("mAP 0.5:0.95", "69.401%")

uploaded_file = st.file_uploader("Choose a ship imagery")
if uploaded_file is not None:
    st.image(uploaded_file, caption='Image to predict')

prediction = st.button("Predict")

if prediction:
    ship_model = torch.hub.load('ultralytics/yolov5', 'custom', path="google colab/YOLOv5/runs/train/exp17/weights/best.pt", force_reload=True)
    results = ship_model(f"C:/Users/bilva/YOLOv5/yolov5/ship_test/{uploaded_file.name}")
    with st.spinner("Loading..."):
        time.sleep(3.5)
        st.success("Done!")
    st.image(np.squeeze(results.render()))
    results.print()
    # with st.echo():
    #     st.text(f"results.print()")
    # st.markdown(results.print())
    # for percent_progress in range (100):
    #     time.sleep(0.1)
    #     progress.progress(percent_progress + 1)
