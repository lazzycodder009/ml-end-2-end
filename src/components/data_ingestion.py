import sys
import os
import pandas as pd
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from src.components.data_transformer import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifact','train.cvs')    
    test_data_path : str = os.path.join('artifact','test.cvs')
    raw_data_path : str = os.path.join('artifact','data.cvs')

class DataIngestion:

    def __init__(self):
        self.ingestion_config= DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:

            df = pd.read_csv('/Users/manish.upadhyay/WorkSpace/ml-end-2-end/data/student.csv')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info(" Data ingestion completed")
            return(
                self.ingestion_config.test_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':
    obj = DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()
    transformer = DataTransformation()
    train_array,test_array,_=transformer.initate_data_transformation(train_path,test_path)
    model_trainer = ModelTrainer()
    model_score = model_trainer.initate_model_trainer(train_array,test_array)
    print(model_score)