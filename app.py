import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect


########################################
# Begin database stuff

# The connect function checks if there is a DATABASE_URL env var.
# If it exists, it uses it to connect to a remote postgres db.
# Otherwise, it connects to a local sqlite db stored in predictions.db.
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################

########################################
# Input validation functions

def check_observation_id(request):
    
    if "observation_id" not in request:
        error = "Field `observation_id` missing from request: {}".format(request)
        return False, error
    
    return True, ""

def check_data(request):
    
    if "data" not in request:
        error = "Field `data` missing from request: {}".format(request)
        return False, error
    
    return True, ""
    
def check_columns(observation):

    valid_columns = {
        'age',
        'workclass',
        'sex',
        'race',
        'education',
        'marital-status',
        'capital-gain',
        'capital-loss',
        'hours-per-week'
    }
    
    keys = set(observation.keys())

    
    if len(valid_columns - keys) > 0: 
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        return False, error
    
    if len(keys - valid_columns) > 0: 
        extra = keys - valid_columns
        error =  "Extra columns: {}".format(extra)
        return False, error    

    return True, ""

def check_categorical(observation):
    
    valid_category_map = {
        "sex": [
            "Male",
            "Female"
        ],
        "race": [
            'White', 
            'Black',
            'Asian-Pac-Islander',
            'Amer-Indian-Eskimo',
            'Other'
        ],
        "workclass": [
            'State-gov', 
            'Self-emp-not-inc',
            'Private',
            'Federal-gov',
            'Local-gov',
            '?', 
            'Self-emp-inc',
            'Without-pay',
            'Never-worked'
        ],
        "education": [
            'Bachelors',
            'HS-grad',
            '11th',
            'Masters',
            '9th',
            'Some-college',
            'Assoc-acdm',
            'Assoc-voc',
            '7th-8th',
            'Doctorate',
            'Prof-school',
            '5th-6th',
            '10th',
            '1st-4th',
            'Preschool',
            '12th'
        ],
        "marital-status": [
            'Never-married',
            'Married-civ-spouse',
            'Divorced',
            'Married-spouse-absent',
            'Separated',
            'Married-AF-spouse',
            'Widowed'
        ]

    }
    
    for key, valid_categories in valid_category_map.items():
        if key in observation:
            value = observation[key]
            if value not in valid_categories:
                error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                    key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
                return False, error
        else:
            error = "Categorical field {} missing".format(key)
            return False, error

    return True, ""

def check_age(observation):
    """
        Validates that observation contains valid hour value 
        
        Returns:
        - assertion value: True if hour is valid, False otherwise
        - error message: empty if hour is valid, False otherwise
    """
    
    age = observation.get("age")
        
    if not age: 
        error = "Field `age` missing"
        return False, error

    if not isinstance(age, int):
        error = "Field `age` = {} is not an integer".format(age)
        return False, error
    
    if age < 0 or age > 100:
        error = "Field `age` = {} is not between 10 and 100".format(age)
        return False, error

    return True, ""
    # YOUR CODE HERE

def check_capital_gain(observation):
    
    capital_gain = observation.get("capital-gain")
        
    # if not capital_gain:
    #     error = "Field `capital-gain` missing"
    #     return False, error

    if not isinstance(capital_gain, int):
        error = "Field `capital-gain`= {} is not an integer".format(capital_gain)
        return False, error
    
    if capital_gain < 0 or capital_gain > 100000:
        error = "Field `capital-gain` = {} is not between 0 and 100,000".format(capital_gain)
        return False, error

    return True, ""


def check_capital_loss(observation):
    
    capital_loss = observation.get("capital-loss")
        
    # if not capital_loss:
    #     error = "Field `capital-loss` missing"
    #     return False, error

    if not isinstance(capital_loss, int):
        error = "Field `capital-loss`= {} is not an integer".format(capital_loss)
        return False, error
    
    if capital_loss < 0 or capital_loss > 100000:
        error = "Field `capital-loss` = {} is not between 0 and 100,000".format(capital_loss)
        return False, error

    return True, ""
    
    
def check_hours_per_week(observation):
    
    hours_per_week = observation.get("hours-per-week")
        
    if not hours_per_week:
        error = "Field `hours-per-week` missing"
        return False, error

    if not isinstance(hours_per_week, int):
        error = "Field `hours-per-week` = {} is not an integer".format(hours_per_week)
        return False, error
    
    if hours_per_week < 0 or hours_per_week > 168:
        error = "Field `hours-per-week` = {} is not between 0 and 168".format(hours_per_week)
        return False, error

    return True, ""
    
# End input validation functions
########################################

########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    obs_dict = request.get_json()
  
    observation_id_ok, error = check_observation_id(obs_dict)
    if not observation_id_ok:
        response = {'error': error}
        return response
    
    _id = obs_dict['observation_id']
    
    
    
    
    data_ok, error = check_data(obs_dict)
    if not data_ok:
        response = {'error': error}
        return response
    
    observation = obs_dict['data']
    
    
    columns_ok, error = check_columns(observation)
    if not columns_ok:
        response = {'error': error}
        return response
    
    categorical_ok, error = check_categorical(observation)
    if not categorical_ok:
        response = {'error': error}
        return response
    
    age_ok, error = check_age(observation)
    if not age_ok:
        response = {'error': error}
        return response
    
    capital_gain_ok, error = check_capital_gain(observation)
    if not capital_gain_ok:
        response = {'error': error}
        return response

    capital_loss_ok, error = check_capital_loss(observation)
    if not capital_loss_ok:
        response = {'error': error}
        return response
    
    hours_per_week_ok, error = check_hours_per_week(observation)
    if not hours_per_week_ok:
        response = {'error': error}
        return response
    
    
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    proba = pipeline.predict_proba(obs)[0, 1]
    prediction = pipeline.predict(obs)[0]
    
    response = {'observation_id': _id}
    response.update(observation)
    response.update({'prediction': bool(prediction), 'probability': proba})

    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data,
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)



# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
