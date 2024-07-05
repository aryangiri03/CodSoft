from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Function to preprocess data
def preprocess_data(df):
    df.dropna(subset=['Age', 'Embarked'], inplace=True)
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = df['Embarked'].astype('category')
    emb_dummy = pd.get_dummies(df['Embarked'])
    emb_dummy = emb_dummy.astype(int)
    df.reset_index(drop=True, inplace=True)
    emb_dummy.reset_index(drop=True, inplace=True)
    df = pd.concat([df, emb_dummy], axis=1)
    df.drop(columns=['Embarked'], inplace=True)
    return df

# Function to create heatmap
def create_heatmap(df):
    r_df = df.drop(columns=['Name', 'Ticket', 'Cabin'], axis=1)
    plt.figure(figsize=(10, 5))
    r = r_df.corr()
    sns.heatmap(r, annot=True, cmap=plt.cm.CMRmap_r)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return image

# Function to train model
def train_model(df):
    X = df.drop(['Survived', 'PassengerId', 'Name', 'Age', 'Ticket', 'Cabin'], axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = KNeighborsClassifier(n_neighbors=8)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)
    return model, scaler, score, report, matrix, y_test, y_pred

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        df = pd.read_csv(file)
        df = preprocess_data(df)
        heatmap = create_heatmap(df)
        model, scaler, score, report, matrix, y_test, y_pred = train_model(df)
        joblib.dump(model, 'titanicModel.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        
        results_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })
        correct_predictions = (results_df['Actual'] == results_df['Predicted']).sum()
        wrong_predictions = (results_df['Actual'] != results_df['Predicted']).sum()
        
        return render_template('results.html', heatmap=heatmap, score=score, report=report, matrix=matrix, 
                               correct_predictions=correct_predictions, wrong_predictions=wrong_predictions, 
                               results_df=results_df.to_html())
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
