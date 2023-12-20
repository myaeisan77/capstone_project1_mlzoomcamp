import pickle

from flask import Flask
from flask import request
from flask import jsonify
from sklearn import preprocessing 



def load(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)


#dv = load('/home/administrator/mlzoomcamp/capstone_project_1/dv.bin')

model = load('/home/administrator/mlzoomcamp/capstone_project_1/model.bin')

app = Flask('predict')


def predict():
    client = request.get_json()
    label_encoder = preprocessing.LabelEncoder()
    X = label_encoder.fit_transform([client])
    y_pred = model.predict(X,model)
    #get_card = y_pred >= 0.5

    result = {
        'get_probability': float(y_pred),
       # 'get_card': bool(get_card)
    }

    return jsonify(result)


#@app.route('/test',methods = ['GET'])

#def test():
    #url = "http://localhost:9697/predict"
    #client = {"	fixed_acidity": 7.4, "residual_sugar": 1.9, "density": 0.9978}
    #response = requests.post(url, json=client).json()
    #print(response)
    #return 'test'




if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9697)