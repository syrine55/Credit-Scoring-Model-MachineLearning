import pandas as pd
from pandas.core.algorithms import mode
import numpy as np
from flask import Flask, jsonify, request, render_template, make_response
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json

import logging
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
model = CatBoostClassifier()
model.load_model('FinalModelCatBoost')
col_to_norm = ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5',
                   'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 
                   'BILL_TOT', 'PAY_AMT_TOT', 'REMAINING_PAY_BALANCE']

def Normalization(X, scaler="minmax"):
    if scaler.upper() == "STANDARD":
        stand = StandardScaler()
        X_s = stand.fit_transform(X)
        return X_s
    else:
        minmax = MinMaxScaler()
        X_mm = minmax.fit_transform(X)
        return X_mm

# app
app = Flask(__name__)

# routes


@app.route('/predict', methods=['POST'])
def predict():
    # get data
    app.logger.warning("dqdqsdqssdsqdqsdsqqdds")

    app.logger.warning(request.form.values())

    # data = request.get_json(force=True)
    # data = json.loads(request.get_data())
    # app.logger.warning(type(data))
    data = request.form
    array = []
    # array = [ 20000,  25238,  25274,  25026,  23073,  22857,  21143,   2000,
    #      1700,      0,   1653,      0,   1940, 142611,   7293, 135318,
    #         0,      0,      0,      0,      1,      0,      0,      0,
    #         0,      0,      0,      0,      0,      0,      0,      1,
    #         0,      0,      0,      0,      0,      0,      0,      0,
    #         0,      1,      0,      0,      0,      0,      0,      0,
    #         0,      0,      0,      0,      1,      0,      0,      0,
    #         0,      0,      0,      0,      0,      0,      1,      0,
    #         0,      0,      0,      0,      0,      0,      0,      0,
    #         1,      0,      0,      0,      0,      0,      0]
    array.append(int(data['LIMIT_BAL']))
    array.append(int(data['BILL_AMT1']))  # bill ammount
    array.append(int(data['BILL_AMT2']))
    array.append(int(data['BILL_AMT3']))
    array.append(int(data['BILL_AMT4']))
    array.append(int(data['BILL_AMT5']))
    array.append(int(data['BILL_AMT6']))
    array.append(int(data['PAY_AMT1'])) # pay ammount
    array.append(int(data['PAY_AMT2']))
    array.append(int(data['PAY_AMT3']))
    array.append(int(data['PAY_AMT4']))
    array.append(int(data['PAY_AMT5']))
    array.append(int(data['PAY_AMT6']))
    bill_tot = int(data['BILL_AMT1']) + int(data['BILL_AMT2']) + int(data['BILL_AMT3']) + int(data['BILL_AMT4']) + int(data['BILL_AMT5']) + int(data['BILL_AMT6'])
    array.append(bill_tot)  # BILL_TOT
    pay_amt_tot = int(data['PAY_AMT1']) + int(data['PAY_AMT2']) + int(data['PAY_AMT3']) + int(data['PAY_AMT4']) + int(data['PAY_AMT5']) + int(data['PAY_AMT6'])
    array.append(pay_amt_tot)  # PAY_AMT_TOT
    array.append(bill_tot - pay_amt_tot)  # REMAINING_PAY_BALANCE

    for i in np.arange(-2, 9):
        if (i == int(data['PAY_1'])):
            array.append(1)
        else:
            array.append(0)

    for i in np.arange(-2, 8):
        if (i == int(data['PAY_2'])):
            array.append(1)
        else:
            array.append(0)

    for i in np.arange(-2, 9):
        if (i == int(data['PAY_3'])):
            array.append(1)
        else:
            array.append(0)


    for i in np.arange(-2, 9):
        if (i == int(data['PAY_4'])):
            
            array.append(1)
        else:
            array.append(0)


    for i in np.arange(-2, 9):
        if (i == 1):
            app.logger.warning("do nothing")
        elif (i == int(data['PAY_5'])):
            
            array.append(1)
        else:
            array.append(0)


    for i in np.arange(-2, 9):
        if (i == 1):
            app.logger.warning("do nothing")
        elif (i == int(data['PAY_6'])):
            array.append(1)
        else:
            array.append(0)
    app.logger.warning(len(array))
    
    app.logger.warning(array)
    # columns = ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'BILL_TOT', 'PAY_AMT_TOT', 'REMAINING_PAY_BALANCE', 'PAY_1__-2', 'PAY_1__-1', 'PAY_1__0', 'PAY_1__1', 'PAY_1__2', 'PAY_1__3', 'PAY_1__4', 'PAY_1__5', 'PAY_1__6', 'PAY_1__7', 'PAY_1__8', 'PAY_2__-2', 'PAY_2__-1', 'PAY_2__0', 'PAY_2__1', 'PAY_2__2', 'PAY_2__3', 'PAY_2__4', 'PAY_2__5', 'PAY_2__6', 'PAY_2__7', 'PAY_3__-2', 'PAY_3__-1', 'PAY_3__0', 'PAY_3__1', 'PAY_3__2', 'PAY_3__3', 'PAY_3__4', 'PAY_3__5', 'PAY_3__6', 'PAY_3__7', 'PAY_3__8', 'PAY_4__-2', 'PAY_4__-1', 'PAY_4__0', 'PAY_4__1', 'PAY_4__2', 'PAY_4__3', 'PAY_4__4', 'PAY_4__5', 'PAY_4__6', 'PAY_4__7', 'PAY_4__8', 'PAY_5__-2', 'PAY_5__-1', 'PAY_5__0', 'PAY_5__2', 'PAY_5__3', 'PAY_5__4', 'PAY_5__5', 'PAY_5__6', 'PAY_5__7', 'PAY_5__8', 'PAY_6__-2', 'PAY_6__-1', 'PAY_6__0', 'PAY_6__2', 'PAY_6__3', 'PAY_6__4', 'PAY_6__5', 'PAY_6__6', 'PAY_6__7', 'PAY_6__8']
    app.logger.info(len(model.feature_names_))    
    app.logger.warning(len(array))

    # dataframe = pd.DataFrame(array, columns)




    # app.logger.warning(len(array))
    # stand = StandardScaler()
    # stand.fit(dataframe)
    # normalizedData = stand.transform(dataframe)
    # normalizedDataArray = []
    # for i in np.arange(0, len(normalizedData)):
    #     normalizedDataArray.append(normalizedData[i][0])
    app.logger.info(array)

    # testtttt = [-1.2219697031279304,
    # -0.33716930809891577,
    # -0.3155274955619973,
    # -0.29326421586605583,
    # -0.28434679523592227,
    # -0.2499018063299386,
    # -0.26712687742695224,
    # -0.24630073712442627,
    # -0.2565088859889257,
    # -0.4524108745841818,
    # -0.2258722754809497,
    # -0.3959608904279629,
    # -0.17962199901573084,
    # -0.3085072943767841,
    # -0.5616668715534848,
    # -0.2606611655627834,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 1.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 1.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 1.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 1.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 1.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 1.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0,
    # 0.0]
    prediction = model.predict(array)
    app.logger.info(prediction)
    # acc_pred_catboost = accuracy_score([1],prediction)
    # r = make_response( int(prediction) )
    # r.mimetype = 'application/json'
    # return r
    # pd.DataFrame(array)
    return render_template('index.html', prediction_text='default payment next month should be = {}'.format(prediction))


@app.route('/pred', methods=['GET'])
def predict2():

    # return data
    []
    return jsonify(model.feature_names_)


@app.route('/', methods=['GET'])
def Home():

    # return data
    []
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)



