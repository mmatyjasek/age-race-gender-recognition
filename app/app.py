#Library imports
import numpy as np
import pandas as pd
import streamlit as st
import cv2
from keras.models import load_model


# Loading the best models
model_age3 = load_model('tuned_model2_age3.h5')
model_age7 = load_model('tuned_model3_age7.h5')
model_gender = load_model('tuned_model1_gender.h5')
model_race = load_model('tuned_model1_race.h5')

# Name of Classes
CLASS_NAMES_AGE3 = ['0: 0 - 17 YO', '1: 18 - 60 YO', '2: 61 YO and above']
CLASS_NAMES_AGE7 = ['0: 0 - 2 YO', '1: 3 - 9 YO', '2: 10 - 17 YO', '3: 18 - 34 YO', '4: 35 - 59 YO', '5: 60 - 79 YO', '6: 80 YO and above']
CLASS_NAMES_GENDER = ['0: male', '1: female']
CLASS_NAMES_RACE = ['0: white', '1: black', '2: Asian', '3: Indian', '4: others']

# Setting Title of App
st.title("Age group, gender and race prediction")
st.markdown("Upload the face image")

# Uploading the dog image
face_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Process')
# On 'Process' button click
if submit:

    if face_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(face_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

        # Displaying the image
        st.image(opencv_image_rgb, channels="RGB")
        
        # Resizing the image
        opencv_image_rgb = cv2.resize(opencv_image_rgb, (200, 200))/255
        
        # Convert image to 4 Dimension
        opencv_image_rgb.shape = (1,200,200,3)
        # Make Prediction
        age3_pred = model_age3.predict(opencv_image_rgb)
        age7_pred = model_age7.predict(opencv_image_rgb)
        gender_pred = model_gender.predict(opencv_image_rgb)
        race_pred = model_race.predict(opencv_image_rgb)

       
        #  Display results in a table
        results = {'3 AGE GROUPS': CLASS_NAMES_AGE3[np.argmax(age3_pred)],
                   '7 AGE GROUPS': CLASS_NAMES_AGE7[np.argmax(age7_pred)],
                   'GENDER': CLASS_NAMES_GENDER[(gender_pred[0] > 0.5).astype("int32")[0]],
                   'RACE': CLASS_NAMES_RACE[np.argmax(race_pred)]}
        df = pd.DataFrame(data=results, index=['RESULTS'])
        st.table(df)
                   
