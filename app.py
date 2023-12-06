from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__, template_folder='./Templates/')
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def view_prediction():
    print('hello')
    if request.method == 'POST':
        no_times_pregnant = int(request.form['pregnancies'])
        glucose_concentration  = int(request.form['glucose'])
        blood_pressure = int(request.form['bloodpressure'])
        skin_fold_thickness = int(request.form['skinthickness'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree = float(request.form['dpf'])
        age = int(request.form['age'])

        data = np.array([[no_times_pregnant, glucose_concentration, blood_pressure, skin_fold_thickness, bmi, diabetes_pedigree, age]])
        # print(data)
        my_prediction = model.predict(data)

        return render_template('prediction.html', prediction = my_prediction)



if __name__ == '__main__':
    app.run(debug=True)