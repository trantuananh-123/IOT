from flask import Flask, request, render_template, redirect
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from database import db
from RandomForest.RandomForest import RandomForest
from RandomForest.Utils import accuracy_score
from SVM.SVM import SVM
from LogisticRegression.LogisticRegression import LogitRegression

import pickle
import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost/heart'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

with app.app_context():
    sql = "SELECT * FROM heart_attack"
    df = pd.read_sql(sql, db.engine)

    X = df.drop(['HeartDisease', 'index'], axis=1)
    y = df['HeartDisease']


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    model = None
    if request.form:

        data = request.form
        type = data['type']
        model_name = data['Model_Name']
        if model_name == '1':
            filename = 'D:/Nam-4/Nam-4/IoT/Project/BTL/model/LogisticRegressionModel.sav'
        elif model_name == '2':
            filename = 'D:/Nam-4/Nam-4/IoT/Project/BTL/model/RandomForestModel.sav'
        elif model_name == '3':
            filename = 'D:/Nam-4/Nam-4/IoT/Project/BTL/model/SupportVectorMachineModel.sav'

        if not os.path.isfile(filename):
            return render_template("index.html", error='Model không tồn tại')
        else:
            BMI = float(data['BMI'])
            Smoking = float(data['Smoking'])
            Alcohol = float(data['Alcohol'])
            Stroke = float(data['Stroke'])
            Physical_Health = float(data['Physical_Health'])
            Mental_Health = float(data['Mental_Health'])
            Difficulty_Walking = float(data['Difficulty_Walking'])
            Sex = float(data['Sex'])
            Age_group = float(data['Age_Group'])
            Diabetes = float(data['Diabetes'])
            Physical_Activity = float(data['Physical_Activity'])
            General_Health = float(data['General_Health'])
            Asthma = float(data['Asthma'])
            Kidney_Disease = float(data['Kidney_Disease'])
            Skin_Cancer = float(data['Skin_Cancer'])

            predict_data = np.array([[BMI, Smoking, Alcohol, Stroke, Physical_Health, Mental_Health, Difficulty_Walking,
                                      Sex, Age_group, Diabetes, Physical_Activity, General_Health, Asthma,
                                      Kidney_Disease, Skin_Cancer]])

            with open(filename, 'rb') as f:
                model = pickle.load(f)
            # model = pickle.load(open(filename, 'rb'))
            if model.X is not None:
                predict_data = predict_data[0, [model.X.columns.get_loc(c) for c in model.X.columns]]
            prediction = model.predict(np.array([predict_data]))
            print(prediction)

        req = {
            "Model_Name": model_name,
            "BMI": str(BMI),
            "Smoking": str(Smoking),
            "Alcohol": str(Alcohol),
            "Stroke": str(Stroke),
            "Physical_Health": str(Physical_Health),
            "Mental_Health": str(Mental_Health),
            "Difficulty_Walking": str(Difficulty_Walking),
            "Sex": str(Sex),
            "Age_group": str(Age_group),
            "Diabetes": str(Diabetes),
            "Physical_Activity": str(Physical_Activity),
            "General_Health": str(General_Health),
            "Asthma": str(Asthma),
            "Kidney_Disease": str(Kidney_Disease),
            "Skin_Cancer": str(Skin_Cancer)
        }

    return render_template("index.html", tab_type=type, value=prediction, req=req)


@app.route("/train", methods=['POST'])
def train():
    global X, y
    model = None
    y_new = None
    if request.form:

        data = request.form
        type = data['type']
        model_name = data['Model_Name']
        features = data.getlist('Features')

        X_new = df[features]

        if model_name == '3':
            y_new = np.where(y == 0, -1, 1)
        else:
            y_new = y

        X_train, X_test, y_train, y_test = train_test_split(
            X_new, y_new, test_size=0.2, random_state=0)

        if model_name == '1':
            filename = 'D:/Nam-4/Nam-4/IoT/Project/BTL/model/LogisticRegressionModel.sav'
            model = LogitRegression()
            model.fit(X_train, y_train)
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
        elif model_name == '2':
            filename = 'D:/Nam-4/Nam-4/IoT/Project/BTL/model/RandomForestModel.sav'
            model = RandomForest()
            model.fit(X_train, y_train)
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
        elif model_name == '3':
            filename = 'D:/Nam-4/Nam-4/IoT/Project/BTL/model/SupportVectorMachineModel.sav'
            model = SVM()
            model.fit(X_train, y_train)
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
    return render_template("index.html", tab_type='accurancy')


@app.route("/accurancy", methods=['POST'])
def accurancy():
    global X, y
    predictions_1 = None
    predictions_2 = None
    y_new = None
    model = None
    cls = None
    if request.form:

        data = request.form
        type = data['type']
        model_name = data['Model_Name']

        if model_name == '3':
            y_new = np.where(y == 0, -1, 1)
        else:
            y_new = y

        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.2, random_state=0)

        if model_name == '1':
            filename = 'D:/Nam-4/Nam-4/IoT/Project/BTL/model/LogisticRegressionModel.sav'
            with open(filename, 'rb') as f:
                model = pickle.load(f)
            X_new = model.X if model.X is not None and model.X.shape[1] != X.shape[1] else X
            y_new = model.y if model.X is not None and model.X.shape[1] != X.shape[1] else y_new
            X_train, X_test, y_train, y_test = train_test_split(
                X_new, y_new, test_size=0.2, random_state=0)

            predictions_1 = model.predict(X_test)

            cls = LogisticRegression(random_state=0)
            cls.fit(X_train, y_train)
            predictions_2 = cls.predict(X_test)
        elif model_name == '2':
            filename = 'D:/Nam-4/Nam-4/IoT/BTL/RandomForestModel.sav'
            with open(filename, 'rb') as f:
                model = pickle.load(f)
            X_new = model.X if model.X is not None and model.X.shape[1] != X.shape[1] else X
            y_new = model.y if model.X is not None and model.X.shape[1] != X.shape[1] else y_new
            X_train, X_test, y_train, y_test = train_test_split(
                X_new, y_new, test_size=0.2, random_state=0)

            predictions_1 = model.predict(X_test)

            cls = RandomForestClassifier(n_estimators=25, max_depth=10, criterion='entropy', random_state=0)
            cls.fit(X_train, y_train)
            predictions_2 = cls.predict(X_test)
        elif model_name == '3':
            filename = 'D:/Nam-4/Nam-4/IoT/Project/BTL/model/SupportVectorMachineModel.sav'
            with open(filename, 'rb') as f:
                model = pickle.load(f)
            X_new = model.X if model.X is not None and model.X.shape[1] != X.shape[1] else X
            y_new = model.y if model.X is not None and model.X.shape[1] != X.shape[1] else y_new
            X_train, X_test, y_train, y_test = train_test_split(
                X_new, y_new, test_size=0.2, random_state=0)

            predictions_1 = model.predict(X_test)

            cls = LinearSVC(random_state=0)
            cls.fit(X_train, y_train)
            predictions_2 = cls.predict(X_test)

    result_1 = accuracy_score(y_test, predictions_1)
    result_2 = accuracy_score(y_test, predictions_2)
    return render_template("index.html", tab_type=type, My_Accurancy=str(result_1 * 100) + '%',
                           Sklearn_Accurancy=str(result_2 * 100) + '%')


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
