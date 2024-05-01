import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, render_template, request, send_from_directory ,request,jsonify
from werkzeug.utils import secure_filename
import numpy as np
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions


app = Flask(__name__,template_folder='')

prediction_done = False

@app.route('/')
def index():
    return render_template('index.html')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filename_utf8 = filename.encode('utf-8').decode('utf-8')  # Encode and then decode to ensure it's a string
        file_path = os.path.join("uploads", filename_utf8)  # Construct file path using os.path.join()
        file.save(file_path)

        # Load pre-trained MobileNetV2 model
        model = MobileNetV2(weights='imagenet')
        # Load and preprocess the image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Perform inference
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=6)[0]  # Get top 6 predictions

        # Initialize an empty list to store top 6 predictions with accuracies
        top_predictions = []

        # Process top 6 decoded predictions
        for label, class_label, probability in decoded_predictions:
            # Convert probability to float
            probability = float(probability)
            
            # Add the prediction to the list
            top_predictions.append((class_label, probability))

        # Render the HTML template and pass the predictions
        return render_template('prediction.html', predictions=top_predictions)

# import pathlib
# from flask import send_from_directory, request

# def display():
#     folder_path = 'runs/detect'
#     directory = pathlib.Path(folder_path)
#     subfolders = [f for f in directory.iterdir() if f.is_dir()]
    
#     if not subfolders:
#         return "No images found"

#     latest_subfolder = max(subfolders, key=lambda x: x.stat().st_ctime)
#     files = list(latest_subfolder.iterdir())
    
#     if not files:
#         return "No images found"

#     latest_file = max(files, key=lambda x: x.stat().st_ctime)

#     filename = str(latest_file)

#     file_extension = filename.rsplit('.', 1)[1].lower()

#     if file_extension in {'jpg', 'jpeg', 'png'}:
#         return send_from_directory(str(latest_subfolder), latest_file.name, environ=request.environ)
#     else:
#         return "Invalid file format"

if __name__ == '__main__':
    app.run(debug=True)