import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
from keras.models import load_model


model = load_model('/content/version3.h5')
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger', 14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple','Banana','Grapes','Kiwi','Lemon','Mango','Orange','Pear','Pineapple','Pomegranate','Watermelon']
vegetables = ['Beetroot','Bello Pepper','Chilli Pepper','Jalepeno','Cabbage','Paprika','Capsicum','Carrot','Cauliflower','Corn','Cucumber','Eggplant','Ginger','Lettuce','Onion','Peas','Potato','Raddish','Soy Beans','Spinach','Sweetcorn','Sweetpotato','Tomato','Turnip']
calNum = {'Apple': 52, 'Banana': 89, 'Beetroot': 43, 'Bell pepper': 26, 'Cabbage': 24, 'Capsicum': 26, 'Carrot': 41, 'Cauliflower': 25, 'Chilli pepper': 40, 'Corn': 86, 'Cucumber': 12, 'Eggplant': 24, 'Garlic': 149, 'Ginger': 80, 'Grapes': 69, 'Jalepeno': 30, 'Kiwi': 61, 'Lemon': 29, 'Lettuce': 14,
          'Mango': 65, 'Onion': 42, 'Orange': 47, 'Paprika': 289, 'Pear': 58, 'Peas': 81, 'Pineapple': 48, 'Pomegranate': 68, 'Potato': 104, 'Raddish': 16, 'Soy beans': 81, 'Spinach': 23, 'Sweetcorn': 86, 'Sweetpotato': 114, 'Tomato': 18, 'Turnip': 28, 'Watermelon': 30}

def processed_uploadedImg(img_path):
    img=load_img(img_path,target_size=(224,224,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    return res.capitalize()

def processed_WebcamImg(img_arr):
    img=np.expand_dims(img_arr/255,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    return res.capitalize()



def run():
    st.title("🥝🍉🍇🥕🥦🌽")
    st.header("What are the Calories of Fruits or Vegetable?")
    st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)
    img_file = st.file_uploader("⬆️ Choose an Image", type=["jpg", "png", "jpeg"])
    img_file_buffer = st.camera_input("📷 Take a picture")

    if img_file_buffer is not None:
        img = Image.open(img_file_buffer).resize((224,224))
        img_array = np.array(img)
        with open(img_file_buffer.name, "wb") as f:
            f.write(img_file_buffer.getbuffer())

        if st.button("Predict"):
            result= processed_WebcamImg(img_array)
            print(result)
            if result in vegetables:
                st.info('**Category : Vegetables**')
            else:
                st.info('**Category : Fruits**')
            st.success("**Predicted : "+result+'**')
            cal = 0
            for key, value in calNum.items():
              if(result == key):
                cal = value
                break
            st.warning('**Calories : '+str(cal)+' Cal / 100 grams**')

    elif img_file is not None:
        img = Image.open(img_file).resize((250,250))
        st.image(img,use_column_width=False)
        save_image_path = img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if st.button("Predict"):
            result= processed_uploadedImg(save_image_path)
            print(result)
            if result in vegetables:
                st.info('**Category : Vegetables**')
            else:
                st.info('**Category : Fruits**')
            st.success("**Predicted : "+result+'**')
            cal = 0
            for key, value in calNum.items():
              if(result == key):
                cal = value
                break
            st.warning('**Calories : '+str(cal)+' Cal / 100 grams**')
run()