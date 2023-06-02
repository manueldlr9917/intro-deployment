from model_utils import update_model, save_simple_metrics_report, get_model_performance_test_set
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

import logging
import sys
import numpy as np
import pandas as pd

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

logging.info('Loading data...')

data=pd.read_csv('dataset/full_data.csv')

logging.info('Loading model...')

model=Pipeline([
    ('imputer',SimpleImputer(strategy='mean',missing_values=np.nan)),
    ('core model', GradientBoostingRegressor())
])

logging.info('Separating dataset into train and test')
X=data.drop(['worldwide_gross'],axis=1)
Y=data['worldwide_gross']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.35, random_state=42)

logging.info('Setting hyperparameter to tune')

param_tuning={'core model__n_estimators':range(20,301,20)}

grid_search=GridSearchCV(model, param_grid=param_tuning, scoring='r2', cv=5)

logging.info('Starting grid search...')

grid_search.fit(X_train,Y_train)

logging.info('Cross validating with best model')

final_result=cross_validate(grid_search.best_estimator_,X_train,Y_train,return_train_score=True,cv=5)
print(final_result)
train_score=np.mean(final_result['train_score'])
test_score=np.mean(final_result['test_score'])

assert train_score > 0.7
assert test_score > 0.65

logging.info(f'Train score: {train_score}')
logging.info(f'Test score: {test_score}')

logging.info('Updating model...')
update_model(grid_search.best_estimator_)

logging.info('Generating model report...')
validation_score=grid_search.best_estimator_.score(X_test,Y_test)
save_simple_metrics_report(train_score,test_score,validation_score,grid_search.best_estimator_)

Y_test_pred=grid_search.best_estimator_.predict(X_test)
get_model_performance_test_set(Y_test,Y_test_pred)

logging.info('Training finished')