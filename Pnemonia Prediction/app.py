
from flask import Flask, request, render_template 
import sys
import io
import os
import numpy as np
from PIL import Image
import cv2
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model, load_model  # Import load_model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

from flask_cors import CORS
from flask import jsonify

# Initialize the Flask app
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow requests from any origin

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

# Load the model for Brain Tumor
base_model = VGG19(include_top=False, input_shape=(240, 240, 3))
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model_brain_tumor = Model(base_model.inputs, output)

# Load weights for Brain Tumor
model_weights_path = r'D:\Downloads\vgg_unfrozen.h5'
model_brain_tumor.load_weights(model_weights_path)

# Load models for Pneumonia, COVID-19, and Malaria
model_pneumonia = load_model(r'D:\C Added Data\model_weights\vgg19_model_01.h5')
model_covid19 = load_model(r'D:\Downloads\my_model_Covid19.keras')
model_maleria = load_model(r'D:\Downloads\my_model_Maleria.keras')  # New Malaria model

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', result=None, show_back_button=False)

"""@app.route('/predict', methods=['POST'])
def upload():
    try:
        if request.method == 'POST':
            disease = request.form['disease']
            f = request.files['file']

            if f:
                basepath = os.path.dirname(__file__)
                file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
                f.save(file_path)

                # Select model based on disease
                if disease == 'brain_tumor':
                    value = getResult(file_path, model_brain_tumor)
                elif disease == 'pneumonia':
                    value = getResult(file_path, model_pneumonia)
                elif disease == 'covid19':
                    value = getResult(file_path, model_covid19)
                elif disease == 'maleria':  # Added Malaria model handling
                    value = getResult(file_path, model_maleria)

                result = get_className(value, disease)
                
                # Render the template with the prediction result and show back button
                return render_template('index.html', result=result, show_back_button=True)
            else:
                return render_template('index.html', result="No file received", show_back_button=False)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return render_template('index.html', result=f"Error: {str(e)}", show_back_button=False)
"""





@app.route('/predict', methods=['POST'])
def upload():
    try:
        if request.method == 'POST':
            disease = request.form['disease']
            f = request.files['file']

            if f:
                basepath = os.path.dirname(__file__)
                upload_folder = os.path.join(basepath, 'static/uploads')
                if not os.path.exists(upload_folder):
                    os.makedirs(upload_folder)  # Create uploads folder if not exists
                file_path = os.path.join(upload_folder, secure_filename(f.filename))
                f.save(file_path)

                # Select model based on disease
                if disease == 'brain_tumor':
                    value = getResult(file_path, model_brain_tumor)
                elif disease == 'pneumonia':
                    value = getResult(file_path, model_pneumonia)
                elif disease == 'covid19':
                    value = getResult(file_path, model_covid19)
                elif disease == 'maleria':  # Added Malaria model handling
                    value = getResult(file_path, model_maleria)

                result = get_className(value, disease)

                # Render the template with the prediction result, image path, and back button
                # Provide the relative path for the image (from the static folder)
                image_url = f"/static/uploads/{secure_filename(f.filename)}"
                return render_template('index.html', result=result, image_url=image_url, show_back_button=True)
            else:
                return render_template('index.html', result="No file received", show_back_button=False)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return render_template('index.html', result=f"Error: {str(e)}", show_back_button=False)







def get_className(classNo, disease):
    if disease == 'brain_tumor':
        return "Yes Brain Tumor" if classNo == 1 else "No Brain Tumor"
    elif disease == 'pneumonia':
        return "Yes Pneumonia" if classNo == 1 else "No Pneumonia"
    elif disease == 'covid19':
        return "Yes COVID-19" if classNo == 0 else "No COVID-19"
    elif disease == 'maleria':
        return "Yes Malaria" if classNo == 1 else "No Malaria"  # Assuming Malaria is class 1

def getResult(img, model):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')

    # Resize the image according to the model's expected input size
    if model == model_pneumonia:
        image = image.resize((128, 128))  # Resize to 128x128 for pneumonia model
    elif model == model_brain_tumor:
        image = image.resize((240, 240))  # Resize to 240x240 for brain tumor model
    elif model == model_covid19:
        image = image.resize((224, 224))  # Resize to 224x224 for COVID-19 model
    elif model == model_maleria:
        image = image.resize((224, 224))  # Resize to 224x224 for Malaria model

    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)

    # Normalize image if needed (if the models expect normalized input)
    input_img = input_img / 255.0

    result = model.predict(input_img)

    # Debug: Check the model output
    print(f"Model Output: {result}")

    if model == model_covid19 or model == model_maleria:  # Specific handling for binary models
       predicted_class = 1 if result > 0.5 else 0
       return predicted_class
    else:
       result01 = np.argmax(result, axis=1)
       return result01

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
