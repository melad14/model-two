from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
import os
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

model = load_model('./my_model.keras')

@app.route("/predict", methods=["POST"])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        f = request.files['file']
        if f.filename == '':
            return jsonify({"error": "No selected file"}), 400

        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'uploads')
        os.makedirs(upload_path, exist_ok=True)
        filepath = os.path.join(upload_path, f.filename)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        pred = model.predict(x)
        y_pred = np.argmax(pred)
        print("prediction", y_pred)

        index = [
            'Myocardial Infarction Patient',
            'Normal Person',
            'Patient that have History of MI',
            'Patient that have abnormal heartbeat'
        ]


        result = str(index[y_pred])

        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=9000)
