from flask import Flask,render_template,request
from flask_material import Material
import pandas as pd 
import numpy as np 
import joblib


app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/preview')
def preview():
    df = pd.read_csv("data/Iris.csv")
    return render_template("preview.html",df_view = df)

@app.route('/',methods=["POST"])
def analyze():
	result_prediction = ''
	if request.method == 'POST':
		petal_length = request.form['petal_length']
		sepal_length = request.form['sepal_length']
		petal_width = request.form['petal_width']
		sepal_width = request.form['sepal_width']
		model_choice = request.form['model_choice']

		# Clean the data by convert from unicode to float 
		sample_data = [sepal_length,sepal_width,petal_length,petal_width]
		clean_data = [float(i) for i in sample_data]

		m_jlib = joblib.load('data/SVM-model')
		result_prediction = m_jlib.predict([clean_data])[0]
		print(result_prediction)

	return render_template('index.html', petal_width=petal_width,
		sepal_width=sepal_width,
		sepal_length=sepal_length,
		petal_length=petal_length,
		clean_data=clean_data,
		result_prediction=result_prediction,
		model_selected=model_choice)


if __name__ == '__main__':
	app.run(debug=True)