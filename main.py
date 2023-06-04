
import tensorflow as tf
from flask import Flask, request, jsonify
#from watchdog.events import EVENT_TYPE_CREATED
import datetime

app = Flask(__name__)
reconstructed_model = tf.keras.models.load_model('new_model.h5')
@app.route("/predict-rainfall-lstms/", methods=["POST"])
def post():
  posted_data = request.get_json()
  Prec = 0
  Temperature = posted_data['Temperature']
  Humidity = posted_data['Humidity']
  Windspeed = posted_data['Windspeed']
  Pressure= posted_data['Pressure']

  data = [[[Prec, Temperature, Humidity, Windspeed, Pressure] for i in range (6) ]]

  prediction = reconstructed_model.predict(data)[0]

  return jsonify({
      'Prediction': str(prediction[0])
  })

@app.route("/predict-rainfall-lstms/", methods=["GET"])
def get():
  return jsonify({
      'Message': 'Hello World'
  })

if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=5000)
