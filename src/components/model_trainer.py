import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_model
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import(
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor


@dataclass
class ModelTrainerConfig:
    model_file_path:str = os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_trainer(self,train_array,test_array):

        try:
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models:dict = {
                'linear_regressor' : LinearRegression(),
                'adaboost': AdaBoostRegressor(),
                'random_forest': RandomForestRegressor(),
                'gradient_boost': GradientBoostingRegressor(),
                'decision_tree': DecisionTreeRegressor(),
                'knn': KNeighborsRegressor(),
                'xgboost': XGBRegressor(),
                'catboost': CatBoostRegressor()
            }
            params:dict={
                "decision_tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "random_forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "gradient_boost":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "linear_regressor":{},
                 "knn":{},
                "xgboost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "catboost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "adaboost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
            
            best_model_score = max(sorted(model_report.values()))
            logging.info(f" Best model score : {best_model_score}")
            best_model_name = list(model_report.keys())[
                    list(model_report.values()).index(best_model_score)
                ]
            best_model = models[best_model_name]
            logging.info(f" Best model Name : {best_model_name}")
            if best_model_score < 0.6: 
                raise CustomException(" No model found")
            
            save_object(
                file_path=self.model_trainer_config.model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)
