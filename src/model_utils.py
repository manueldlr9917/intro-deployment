from sklearn.pipeline import Pipeline
from joblib import dump
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def update_model(model: Pipeline) -> None:
    dump(model,'model/model.pkl')

def save_simple_metrics_report(train_score: float,test_score: float, validation_score: float,model: Pipeline) -> None:
    with open('report.txt','w') as report_file:
        report_file.write('# Model Pipeline Description')

        for key, value in model.named_steps.items():
            report_file.write(f'### {key}:{value.__repr__()}'+'\n')

        report_file.write(f'### Train score: {train_score}'+'\n')
        report_file.write(f'### Test score: {test_score}'+'\n')
        report_file.write(f'### Validation score: {validation_score}'+'\n')

def get_model_performance_test_set(Y_real: pd.Series,Y_pred: pd.Series) -> None:
    fig,ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(8)
    sns.regplot(x=Y_pred, y=Y_real, ax=ax)
    ax.set_xlabel('Predicted worldwide gross')
    ax.set_ylabel('Real worldwide gross')
    ax.set_title('Behavior of model prediction')
    fig.savefig('prediction_behavior.png')



