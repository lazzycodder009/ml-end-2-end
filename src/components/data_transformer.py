import sys
import os
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
#from src.components.data_ingestion import DataIngestion
@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str = os.path.join('artifact','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            num_columns= ["writing_score", "reading_score"]
            cat_columns =[ "gender", "race_ethnicity","parental_level_of_education","lunch","test_preparation_course",]

            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])           
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder',OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))

            ])
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline,num_columns),
                    ('cat_pipeline',cat_pipeline,cat_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
            save_object(

                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':
    transformer = DataTransformation()
    #ingestor   =  DataIngestion()
    #train_path,test_path = ingestor.initiate_data_ingestion()
    #train_arr,test_arr,preprocessor_path= transformer.initate_data_transformation(train_path=train_path,test_path=test_path)
    #print(train_arr)

