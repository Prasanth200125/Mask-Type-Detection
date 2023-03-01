import numpy as np
import cv2
import streamlit as st
from tensorflow import keras



@st.cache(allow_output_mutation=True)


def detect_faces(frame):
    class_indices = {0:'cloth', 1 :'n95', 2: 'n95v', 3:'nfm', 4: 'srg'}
    modelSaved = keras.models.load_model('Face_Mask_Detection.h5') 
    
    final_image = cv2.resize(frame,(224,224))
    final_image = np.expand_dims(final_image, axis=0) 
    final_image = final_image/255.0

    Predictions =modelSaved.predict(final_image)
    maxindex = int(np.argmax(Predictions))
    finalout = class_indices[maxindex]
    return finalout
   
    
        
def main():
 

    st.title(":mask:Face Mask Type Detector:mask: ")
    st.subheader(" By-:sunglasses:V V Prasanth G:sunglasses:")
    selectbox = st.sidebar.selectbox('Select the type of input', ('None','Upload Image', 'Take A Shot'))
    if selectbox == 'Take A Shot':
        image_file = st.camera_input(label='Take A Shot')
        if image_file is not None:
            file_bytes = np.asarray(bytearray(image_file .read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.text("Original Image")
            st.image(frame)
        if st.button("Recognise"):
            result_img= detect_faces(frame)
            
            st.text(result_img)
    if selectbox == 'Upload Image':
        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if image_file is not None:
            file_bytes = np.asarray(bytearray(image_file .read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.text("Original Image")
            st.image(frame)
        if st.button("Recognise"):
            result_img= detect_faces(frame)
            st.text(result_img)
        
    if selectbox == 'None':
        st.header("Choose any option from sidebar to get started->dropdown")
    
    


if __name__ == '__main__':
    main()