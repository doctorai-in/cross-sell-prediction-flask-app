import pandas as pd 
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Import label encoder 
from sklearn import preprocessing 
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
import os
import shutil 
import joblib

#lin_reg = joblib.load("my_model.pkl")

def house_price(filename, model):
     

    test_df= pd.read_csv(filename)
    # Encode labels in column 'species'. 
    test_df['Gender']= label_encoder.fit_transform(test_df['Gender'])  
    test_df['Vehicle_Damage']= label_encoder.fit_transform(test_df['Vehicle_Damage']) 
    test_df['Vehicle_Age'] = test_df['Vehicle_Age'].map({'< 1 Year':1,'1-2 Year':2,'> 2 Years':3})

    X_test = test_df.drop(columns=['id'])
    result = model.predict_proba(X_test)[:,1]

    print("Predictions:", model.predict_proba(X_test)[:,1])


    #############################################################
    #                   Save Model Prediction                   #
    #############################################################
    df_sub = pd.read_csv('sample_submission.csv')
    df_sub.head()
    df_xgb = df_sub.copy()
    df_xgb['Response'] = result
    df_xgb.head()
    validate_dirs("result")
    df_xgb.to_csv('result/XGBoost_Simple-default_final_sub.csv', index=False )
    df_xgb.to_html('templates/output.html')

    shutil.rmtree("files")
    





    '''# Save test predictions to file
    some_data_prepared = pd.DataFrame(some_data_prepared)
    output = pd.DataFrame({'Id': some_data_prepared.index,'Y Original': some_labels, 'Y predicted':lin_reg.predict(some_data_prepared)})
    #strat_test_set.to_csv('data/train.csv', index=False)
    output.to_csv('files/outputTest.txt', index=False)
    
     
    output.to_html('output.html')'''



def validate_dirs(dir):
    try: 
        if not os.path.exists(dir):
            os.makedirs(dir)  
    except OSError:
        print('Error: Creating directory to store faces')


