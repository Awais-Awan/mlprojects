import os
import sys
from  dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj,evaluate_model



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,Y_train,X_test,Y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbour Regressor" : KNeighborsRegressor(),
                "XGBRegressor" : XGBRegressor(),
                "CatBoosting Regressor" : CatBoostRegressor(),
                "AdaBoost Regressor" : AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse'],
                    # 'splitter': ['best', 'random'],
                    'max_depth': [None, 10, 20],
                    # 'min_samples_split': [2, 5, 10],
                    # 'min_samples_leaf': [1, 2, 4],
                },

                "Random Forest": {
                    'n_estimators': [50, 100],
                    # 'criterion': ['squared_error', 'absolute_error'],
                    'max_depth': [10, 20],
                    # 'min_samples_split': [2, 5],
                    # 'min_samples_leaf': [1, 2],
                    # 'max_features': ['sqrt', 'log2'],
                    # 'bootstrap': [True, False],
                },

                "Gradient Boosting": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1],
                    # 'subsample': [0.8, 1.0],
                    'max_depth': [3, 5],
                    # 'min_samples_split': [2, 5],
                    # 'min_samples_leaf': [1, 2],
                },

                "Linear Regression": {},

                "K-Neighbour Regressor": {
                    'n_neighbors': [3, 5, 7],
                    # 'weights': ['uniform', 'distance'],
                    # 'algorithm': ['auto', 'ball_tree'],
                },

                "XGBRegressor": {
                    'learning_rate': [0.05, 0.1],
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5],
                    # 'subsample': [0.8, 1.0],
                    # 'colsample_bytree': [0.8, 1.0],
                    # 'gamma': [0, 0.1],
                },

                "CatBoosting Regressor": {
                    'depth': [6, 8],
                    'learning_rate': [0.05, 0.1],
                    'iterations': [100, 200],
                    # 'l2_leaf_reg': [3, 5],
                },

                "AdaBoost Regressor": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1],
                    # 'loss': ['linear', 'square', 'exponential'],
                }
            }


            model_report: dict= evaluate_model(X_train,Y_train,X_test,Y_test, models=models, param = params)



            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found!")
            logging.info("Best Model Found on both training and testin dataset")


            save_obj(
                self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )


            predicted = best_model.predict(X_test)
            r2_Score = r2_score(Y_test,predicted)
            return r2_Score



        except Exception as e:
            raise CustomException(e, sys)
