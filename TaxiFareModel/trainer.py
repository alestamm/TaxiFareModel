from TaxiFareModel.encoders import *
from TaxiFareModel.utils import compute_rmse
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from TaxiFareModel.data import get_data, clean_data
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
import joblib
import mlflow



class Trainer():
    def __init__(self, X_train=None, y_train=None, X_test=None, y_test=None, pipeline=None):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = pipeline
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.MLFLOW_URI = "https://mlflow.lewagon.co/"
        self.experiment_name = "[BR][SP][alestamm][TaxiFareModel-v1]"

    def set_pipeline(self):
        model = RandomForestRegressor(n_estimators=100)
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', model)
        ])
        self.mlflow_log_param('model', model)

        return pipe

    def run(self):
        """set and train the pipeline"""
        pipeline = self.set_pipeline()
        self.pipeline = pipeline.fit(self.X_train, self.y_train)


        return pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        self.X_test = X_test
        self.y_test = y_test

        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        self.mlflow_log_metric('rmse', rmse)
        return rmse
    def save_model(self, trained_model):
        """ Save the trained model into a model.joblib file """
        joblib.dump(trained_model, 'model.joblib')

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df_clean = clean_data(df)
    # set X and y
    y = df_clean["fare_amount"]
    X = df_clean.drop("fare_amount", axis=1)
    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    # train
    trainer = Trainer(X_train=X_train, y_train=y_train)
    pipeline = trainer.run()
    # evaluate
    rmse = trainer.evaluate(X_val, y_val)
    # save model
    trainer.save_model(pipeline)
