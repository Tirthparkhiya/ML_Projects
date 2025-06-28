import numpy as np
import pandas as pd
import os
import sys
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
            raise CustomException(e,sys)        
           
def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    try:
        report = {}
        best_model = None
        best_score = float("-inf")
        best_model_name = None

        for model_name in models.keys():
            model = models[model_name]
            para = param.get(model_name, {})

            gs = GridSearchCV(model, para, cv=3, error_score='raise')
            gs.fit(x_train, y_train)

            y_test_pred = gs.predict(x_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score

            # Track best model
            if test_model_score > best_score:
                best_score = test_model_score
                best_model = gs.best_estimator_  # Fitted model
                best_model_name = model_name

        return report, best_model_name, best_model
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)        
    
                