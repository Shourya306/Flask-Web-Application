
from flask import Flask, render_template, request
import pickle
import numpy as np


model = pickle.load(open('linear_model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods = ["POST"])
def predict():
    rd_val = float(request.form['RD'])
    admin_val = float(request.form['Admin'])
    marketing_val = float(request.form['Marketing'])
    ny_val = float(request.form['NY'])
    fl_val = float(request.form['FL'])
    cal_val = float(request.form['Cal'])

    result = model.predict(np.array([rd_val,admin_val,marketing_val,cal_val,fl_val,ny_val]).reshape(1, -1))
    return render_template('result.html',result = result)

if __name__ == '__main__':
    app.run(debug=True)