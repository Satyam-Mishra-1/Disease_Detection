# from flask import Flask, request, render_template
# import sys
# import io
# import os
# import numpy as np
# from PIL import Image
# import cv2
# from werkzeug.utils import secure_filename
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
# from tensorflow.keras.applications.vgg19 import VGG19

# # Initialize the Flask app

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# app = Flask(__name__)

# # Load the model
# base_model = VGG19(include_top=False, input_shape=(240, 240, 3))
# x = base_model.output
# flat = Flatten()(x)
# class_1 = Dense(4608, activation='relu')(flat)
# drop_out = Dropout(0.2)(class_1)
# class_2 = Dense(1152, activation='relu')(drop_out)
# output = Dense(2, activation='softmax')(class_2)
# model_03 = Model(base_model.inputs, output)

# # Load weights (update the path as necessary)
# model_weights_path = 'D:\\Downloads\\vgg_unfrozen.h5'  # Updated path
# model_03.load_weights(model_weights_path)

# # Define routes
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def upload():
#     if request.method == 'POST':
#         f = request.files['file']
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)

#         # Ensure that getResult handles the image path correctly
#         value = getResult(file_path)
#         result = get_className(value)

#         # Ensure proper encoding when rendering the template
#         return render_template('index.html', result=result.encode('utf-8').decode('utf-8'))
#     return None


# def get_className(classNo):
#     if classNo == 0:
#         return "No Brain Tumor"
#     elif classNo == 1:
#         return "Yes Brain Tumor"

# def getResult(img):
#     image = cv2.imread(img)
#     image = Image.fromarray(image, 'RGB')
#     image = image.resize((240, 240))
#     image = np.array(image)
#     input_img = np.expand_dims(image, axis=0)
#     input_img = input_img / 255.0  # Normalize the image if needed
#     result = model_03.predict(input_img)
#     print("Model Output Probabilities:", result)
#     result01 = np.argmax(result, axis=1)
#     return result01


# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)





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

from flask import Flask, request, render_template
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

# Load the model for Pneumonia
model_pneumonia = load_model(r'D:\C Added Data\model_weights\vgg19_model_01.h5')
model_covid19 = load_model(r'D:\Downloads\my_model_Covid19.keras')


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', result=None, show_back_button=False)

"""
@app.route('/predict', methods=['POST'])
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

                result = get_className(value, disease)
                
                # Render the template with the prediction result and show back button
                return render_template('index.html', result=result, show_back_button=True)
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

# def getResult(img, model):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')

    # Resize the image according to the model's expected input size
    if model == model_pneumonia:
        image = image.resize((128, 128))  # Resize to 128x128 for pneumonia model
    else:
        image = image.resize((240, 240))  # Resize to 240x240 for brain tumor model
    
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
    result01 = np.argmax(result, axis=1)
    return result01

def getResult(img, model):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')

    if model == model_pneumonia:
        image = image.resize((128, 128))  # Resize to 128x128 for pneumonia model
    else:
        image = image.resize((240, 240))  # Resize to 240x240 for brain tumor model
    
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)

    # Normalize image if needed (if the models expect normalized input)
    input_img = input_img / 255.0

    result = model.predict(input_img)

    # Debug: Check the model output
    print(f"Model Output: {result}")

    result01 = np.argmax(result, axis=1)
    return result01

"""



# Load the model for COVID-19


@app.route('/predict', methods=['POST'])
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
                elif disease == 'covid19':  # Added COVID-19 model handling
                    value = getResult(file_path, model_covid19)

                result = get_className(value, disease)
                
                # Render the template with the prediction result and show back button
                return render_template('index.html', result=result, show_back_button=True)
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
        return "Yes COVID-19" if classNo == 0 else "No COVID-19"  # Assuming COVID-19 is class 0

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

    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)

    # Normalize image if needed (if the models expect normalized input)
    input_img = input_img / 255.0

    result = model.predict(input_img)

    # Debug: Check the model output
    print(f"Model Output: {result}")

    if model == model_covid19 : 
       predicted_class = 1 if result > 0.5 else 0
       return predicted_class
    else :
       result01 = np.argmax(result, axis=1)
       return result01

    


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
