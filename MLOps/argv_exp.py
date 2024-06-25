# MLFLOW-Basic-Operation

## For Dagshub

import dagshub
dagshub.init(repo_owner='chandula00', repo_name='e19-co544-cricket-analytics-and-prediction', mlflow=True)
# dagshub.init(repo_owner='Bimbara28', repo_name='e19-co544-cricket-analytics-and-prediction', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)