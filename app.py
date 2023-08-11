from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

filename = 'EPRFC.pkl'
classifier = pickle.load(open(filename,'rb'))
model = pickle.load(open('EPRFC.pkl','rb'))

app = Flask(__name__, template_folder= "templates") #template folder

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict_value():
    try:
        Department = int(request.form['Department'])
        Education = int(request.form['Education'])
        Sex = int(request.form['Sex'])
        Recruitment = int(request.form['Recruitment'])
        Trainings= int(request.form['Trainings'])
        Age = int(request.form['Age'])
        PerformanceRating = int(request.form['PerformanceRating'])
        ServiceLength = int(request.form['ServiceLength'])
        Achievements = int(request.form['Achievements'])
        Training_score = int(request.form['Training_score'])
        sum_metric = int(request.form['sum_metric'])
        total_score = int(request.form['total_score'])

        input_features = [ Department, Education, Sex, Recruitment,
       Trainings, Age, PerformanceRating, ServiceLength,
       Achievements, Training_score, sum_metric, total_score]
        features_value = [np.array(input_features)]
        feature_name = ['Department', 'Education', 'Sex', 'Recruitment',
       'Trainings', 'Age', 'PerformanceRating', 'ServiceLength',
       'Achievements', 'Training_score','sum_metric','total_score']

        df = pd.DataFrame(features_value, columns=feature_name)
        output = model.predict(df)

        return render_template('index.html', prediction_text='Promotion Prediction: {:.2f}'.format(output[0]))
    except Exception as e:
        return render_template('index.html', prediction_text='Error: {}'.format(e))

    
if __name__ == "__main__":
    app.run(debug=True)
