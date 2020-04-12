from constants import *
from connectors.sql_client import SQLClient
import pandas as pd

import os
class DBClient:
    """
    This is an ML Flow API Client that wraps the calls to ML FLow Database tables with easy to use functions.
    """
    def __init__(self, server, db, user, password):
        """
        Initializes a client
        :param server:
        :param db:
        :param user:
        :param password:
        """

        password = os.getenv(MLFLOW_PWD) or password
        if password is None or password == '':
            assert False, "Cannot connect to mlflow database. Please provide password."
        self.sql_client = SQLClient(server=server, db=db, user=user, password=password)

    def list_models(self, text=None, uuid=None, tag=None):
        """
        Lists all the models from the db. If text is given, then runs a full-text search on uuid and tag
        :param text:
        :param uuid:
        :param tag:
        :return:
        """
        sql = 'SELECT * FROM ' + SCHEMA_MODEL_STORE + '.' + TBL_MODEL_STORE
        if text is None and tag is None and uuid is None:
            pass
        else:
            sql = sql + ' WHERE '

        if text is not None:
            sql = sql + 'model_tag LIKE  \'%' + text + '%\' OR model_uuid LIKE \'%' + text + '%\''
        if tag is not None:
            sql = sql + 'model_tag LIKE  \'%' + text + '%\''
        if uuid is not None:
            sql = sql + 'model_uuid LIKE \'%' + text + '%\''

        sql = sql + ' ORDER BY created_date DESC'

        models_df = pd.read_sql(sql, self.sql_client.engine)
        return models_df

