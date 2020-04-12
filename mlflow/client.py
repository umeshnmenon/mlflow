from autologging import logged
import logging
import pandas as pd
from datetime import datetime
import zipfile
from sklearn.externals import joblib
import pickle

# from connectors.sql_client import HTTPClient
from utils.log import *
from utils.common import *
from helpers.config_helper import ConfigHelper
from helpers.common_helper import *
from pymodel import *
from api.mlflow_api_client import MLFlowAPIClient

setup_logger(log_level="DEBUG")  # TODO: Make it configurable as a start up param
logging.getLogger().setLevel(logging.INFO)
@logged(logging.getLogger())
class Client:
    """
    This class handles the MLFlow API calls to upload and deploy the models. The class also takes care of evaluation and
    comparison of models in a training pipeline.
    """

    def __init__(self, name=None, email=None, session=None, input_config=None, output_config=None, mlflow_config=None):
        self.__log.info("Initialiizing the Client...")
        self.name = name
        self.email = email
        self.session = session
        self.config = ConfigHelper().get_config()
        self.__log.info("Creating MLFlow API client....")
        #self.httpClient = create_http_client() # This will make HTTP Calls to MlFlow's Orchestration APIs
        #self.sql_client = create_mlflow_client() # This is a work around for MLFLow's Orchestration APIs as APIs are not yet
                                           # implemented. TODO: Remove this when we have the apis ready
        self.api_client = create_mlflow_client()

        self.__log.info("Successfully created MLFlow API client")
        # Input and Output DataClients will create connection to the specified data source to load the data for training
        #  and storing the predictions back respectively.
        if input_config is not None:
            self.__log.info("Creating input connection....")
            self.input_data_client = create_data_client(input_config)
            self.__log.info("Successfully created input connection")

        if output_config is not None:
            self.__log.info("Creating output connection....")
            self.output_data_client = create_data_client(output_config)
            self.__log.info("Successfully created output connection")


    def list_models(self, text=None, uuid=None, tag=None):
        """
        Lists all the models from the db. If text is given, then runs a full-text search on uuid and tag
        :param text:
        :param uuid:
        :param tag:
        :return:
        """
        return self.api_client.list_models(text, uuid, tag)

    def upload_model(self, model):
        """
        Uploads a MLFlow model to Model Repo
        :param model:
        :return:
        """
        self.__log.info("Uploading the model to Model Store....")
        if model is None:
            assert False, "Model must be provided"

        # Upload the model and artifacts
        # TODO: Make this api call
        #try:
        table = self.config.get(MLFLOW, "model_store")
        schema = self.config.get(MLFLOW, "model_store.schema")
        model_store_df = model.metadata.to_table()
        model_store_df["Model"] = model.artifacts
        print(model_store_df)
        model_store_df.to_sql(table, schema=schema, con=self.sql_client.engine, if_exists='append',
                           index=False)
        self.__log.info("Model uploaded to the Model Store successfully.")
        #except Exception as e:
        #    msg = "Error uploading model. Error: {}".format(str(e))
        #    self.__log.error(msg)
        #    assert False, msg

    def verify_model(self, model):
        """
        Verifies the model. Here it first stores the model to LeaderBoard
        :param model:
        :return:
        """
        self.__log("Verifying the model....")
        if model is None:
            assert False, "Model must be provided"

        # Create LeaderBoard dataframe
        data = [{"model_UUID": model.metadata.model_uuid, "created_date": datetime.now(), "created_by": get_username3()}]
        lb_df = pd.DataFrame(data)

        # Upload the model uuid to LeaderBoard
        try:
            table = self.config.get(MLFLOW, "model_leaderboard")
            schema = self.config.get(MLFLOW, "model_leaderboard.schema")
            data.to_sql(table, schema=schema, con=self.sql_client.engine, if_exists='append',
                                  index=False)
            self.__log.info("Model moved to Leader Board successfully.")
        except Exception as e:
            msg = "Error uploading model to Leader Board. Error: {}".format(str(e))
            self.__log.error(msg)
            assert False, msg

    def deploy_model(self, model):
        """
        Deploys the model
        :param model:
        :return:
        """
        pass

    def load_data(self, sql):
        """
        Loads the data from database specified in inputconfig and executes the sql and returns a pandas dataframe
        :param sql:
        :return:
        """
        data = pd.read_sql(sql, self.input_data_client)
        return data

    def download_model(self, model_uuid=None, model_tag=None, model_version=None):
        """
        Downloads the model from the database using either model_uuid or model_tag and version
        :param model_uuid:
        :param model_tag:
        :param model_version:
        :return:
        """
        pymodel = PyModel(model_uuid=model_uuid, model_tag=model_tag, model_version=model_version)
        return pymodel

    def load_model(self, model_tag, model_version=None):
        """
        Loads the model from the database using either model_uuid or model_tag and version and converts to a PyModel
        :param model_uuid:
        :param model_tag:
        :param model_version:
        :return:
        """
        return self._load_model_by_tag(self, model_tag, model_version)

    def load_model(self, model_uuid):
        """
        Loads the model from the database using either model_uuid or model_tag and version and converts to a PyModel
        :param model_uuid:
        :param model_tag:
        :param model_version:
        :return:
        """
        return self._load_model_by_uuid(model_uuid);

    def _load_model_by_uuid(self, model_uuid):
        """
        Loads the model from database for a given model_uuid
        :param model_uuid:
        :return:
        """
        model_df = pd.read_sql(
            'select * from {} where model_uuid = {}'.format(
                TBL_MODEL_STORE,
                model_uuid
            ),
            self.sql_client.engine
        )
        return self.create_pymodel_from_pandas(model_df)

    def _load_model_by_tag(self, model_tag, model_version=None):
        """
        Loads the model from database for a given model_uuid
        :param model_uuid:
        :return:
        """
        sql = 'SELECT TOP 1 * FROM {} WHERE model_tag = {}'
        if model_version is not None:
            sql = sql + ' and model_version = {}'
        sql = sql + 'ORDER BY created_date DESC'
        sql = sql.format(
                TBL_MODEL_STORE,
                model_tag,
                model_version
            )

        model_df = pd.read_sql(sql, self.sql_client.engine)
        return self.create_pymodel_from_pandas(model_df)

    def create_pymodel_from_pandas(self, model_df):
        """
        Creates a PyModel class from a Pandas Dataframe
        :param model_df:
        :return:
        """
        # first extract the model from the binary BLOB data
        # store the binary data to a file
        tmp_dir = tempfile.TemporaryDirectory()
        zip_file_path = tmp_dir + "/" + model_df["model_uuid"] + ".zip"
        model_zip_file = open(zip_file_path, "wb")
        binary_format = bytearray(model_df["model_artifacts"])
        model_zip_file.write(binary_format)
        model_zip_file.close()
        # extract the zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

        model = joblib.load(tmp_dir + "/model.pkl")
        features = pickle.load(open('feature_column.pkl', 'rb'))
        pymodel = PyModel(model=model, features=features, split=model_df["split"],
                          feature_columns=model_df["feature_columns"],
                          target_column=model_df["target_column"], model_uuid=model_df["model_uuid"],
                          model_tag=model_df["model_tag"], model_version=model_df["model_version"],
                          maximising_metric=model_df["maximising_metric"], hyperparams=model_df["hyperparams"],
                          training_args=model_df["training_args"], train_actuals=model_df["train_actuals"],
                          train_pred_probs=model_df["train_pred_probs"], valid_actuals=model_df["valid_actuals"],
                          valid_pred_probs=model_df["valid_pred_probs"])
        return pymodel