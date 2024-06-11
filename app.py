from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
import datetime
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the model
with open('model (1).pkl', 'rb') as file:
    model = pickle.load(file)

# Label encoding function for counter_type
def label_encode_counter_type(df):
    label_encoder = preprocessing.LabelEncoder()
    df['counter_type'] = label_encoder.fit_transform(df['counter_type'])
    return df

# Data preparation functions
def client_feature_eng(client):
    client['creation_date'] = pd.to_datetime(client['creation_date'])
    client['client_catg'] = client['client_catg'].astype('category')
    client['disrict'] = client['disrict'].astype('category')
    client['region'] = client['region'].astype('category')
    client['TSC'] = (2021 - client['creation_date'].dt.year) * 12 - client['creation_date'].dt.month
    return client

def invoice_feature_eng(invoice):
    invoice['invoice_date'] = pd.to_datetime(invoice['invoice_date'], dayfirst=True)
    invoice['invoice_day'] = invoice['invoice_date'].dt.day
    invoice['invoice_week'] = invoice['invoice_date'].dt.isocalendar().week
    invoice['invoice_month'] = invoice['invoice_date'].dt.month
    invoice['invoice_year'] = invoice['invoice_date'].dt.year
    invoice['counter_statue'] = invoice['counter_statue'].map({
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 769: 5, '0': 0, '5': 5, '1': 1, '4': 4, 'A': 0,
        618: 5, 269375: 5, 46: 5, 420: 5
    })
    invoice['delta_index'] = invoice['new_index'] - invoice['old_index']
    invoice['is_weekday'] = ((pd.DatetimeIndex(invoice.invoice_date).dayofweek) // 5 == 1).astype(float)
    return invoice

def agg_feature(invoice, client_df, agg_stat):
    invoice['delta_time'] = invoice.sort_values(['client_id', 'invoice_date']).groupby('client_id')['invoice_date'].diff().dt.days.reset_index(drop=True)
    agg_trans = invoice.groupby('client_id')[agg_stat + ['delta_time']].agg(['median', 'nunique', 'mean', 'std', 'min', 'max'])
    agg_trans.columns = ['_'.join(col).strip() for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)

    df = invoice.groupby('client_id').size().reset_index(name='transactions_count')
    agg_trans = pd.merge(df, agg_trans, on='client_id', how='left')

    weekday_avg = invoice.groupby('client_id')[['is_weekday']].agg(['mean'])
    weekday_avg.columns = ['_'.join(col).strip() for col in weekday_avg.columns.values]
    weekday_avg.reset_index(inplace=True)
    client_df = pd.merge(client_df, weekday_avg, on='client_id', how='left')

    full_df = pd.merge(client_df, agg_trans, on='client_id', how='left')
    full_df['invoice_per_cooperation'] = full_df['transactions_count'] / full_df['TSC']
    return full_df

def new_features(df):
    agg_stat_columns = [
        'tarif_type', 'counter_number', 'counter_statue', 'counter_code',
        'reading_remarque', 'consommation_level_1', 'consommation_level_2',
        'consommation_level_3', 'consommation_level_4', 'old_index',
        'new_index', 'months_number', 'counter_type', 'invoice_month',
        'invoice_year', 'delta_index'
    ]
    for col in agg_stat_columns:
        df[col + '_range'] = df[col + '_max'] - df[col + '_min']
        df[col + '_max_mean'] = df[col + '_max'] / df[col + '_mean']
    return df

def drop(df):
    col_drop = ['client_id', 'creation_date']
    for col in col_drop:
        df.drop([col], axis=1, inplace=True)
    return df

agg_stat_columns = [
    'tarif_type', 'counter_number', 'counter_statue', 'counter_code',
    'reading_remarque', 'consommation_level_1', 'consommation_level_2',
    'consommation_level_3', 'consommation_level_4', 'old_index',
    'new_index', 'months_number', 'counter_type', 'invoice_month',
    'invoice_year', 'delta_index'
]

drop_col = [
    'reading_remarque_max', 'counter_statue_min', 'counter_type_min',
    'counter_type_max', 'counter_type_range', 'tarif_type_max',
    'delta_index_min', 'consommation_level_4_mean'
]

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'client_file' not in request.files or 'invoice_file' not in request.files:
        return "No file part", 400
    
    client_file = request.files['client_file']
    invoice_file = request.files['invoice_file']
    
    if client_file.filename == '' or invoice_file.filename == '':
        return "No selected file", 400
    
    # Read the uploaded CSV files
    client_data = pd.read_csv(client_file)
    invoice_data = pd.read_csv(invoice_file)

    client_data['region_group'] = client_data['region'].apply(lambda x: 100 if x < 100 else 300 if x > 300 else 200)

    # Label encoding
    invoice_data = label_encode_counter_type(invoice_data)

    # Feature engineering
    client_data_prepared = client_feature_eng(client_data)
    invoice_data_prepared = invoice_feature_eng(invoice_data)
    client_data_agg = agg_feature(invoice_data_prepared, client_data_prepared, agg_stat_columns)
    client_data_features = new_features(client_data_agg)
    client_data_final = drop(client_data_features).drop(drop_col, axis=1)


    # Make predictions
    predictions = model.predict_proba(client_data_final)[:, 1]  # Get probabilities for the positive class

    results = []
    for i, client_id in enumerate(client_data['client_id']):
        fraud_detected = "Yes" if predictions[i] > 0.5 else "No"
        certainty = round(predictions[i] , 2)
        results.append({
            'client_id': client_id,
            'fraud_detected': fraud_detected,
            'certainty': certainty
        })

    return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
