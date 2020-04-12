import pandas as pd
from .constants import *
from .helpers.common_helper import *
from .pymodel import *

class OnlineClient:
    """
    This loads a model from the model repository and provisions for scoring
    """

    def __init(self):
        self.sql_client = create_mlflow_client()  # This is a work around for MLFLow's Orchestration APIs as APIs are not yet
        # implemented

    def get_model(self, model_uuid):
        """
        Loads the model from the Model Store for a given UUID
        :param model_uuid:
        :return:
        """
        model_df = pd.read_sql(
            'select * from {} where model_uuid = {}'.format(
                TBL_MODEL_STORE,
                model_uuid
            ),
            self.sql_client
        )


    def create_pymodel_from_pandas(self, model_df):
        """
        Creates a PyModel class from a Pandas Dataframe
        :param model_df:
        :return:
        """
        pymodel = PyModel()
        # TODO