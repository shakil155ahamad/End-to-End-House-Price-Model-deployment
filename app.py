import numpy as np
from sklearn.preprocessing import RobustScaler
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [int(x) for x in request.form.values()]
    features = [np.array(features)]
    rb_scaler=RobustScaler()
    final_features=rb_scaler.fit_transform(features)


    prediction = np.exp(model.predict(final_features))

    output = round(prediction[0], 0)

    return render_template('index.html', prediction_text='Your House Sale Price is: ${}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
