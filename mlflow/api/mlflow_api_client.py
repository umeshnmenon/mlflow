from api.api_client import APIClient
from api.db_client import DBClient

class MLFlowAPIClient:
    """
    ML Flow API Client wraps the calls to ML FLow API with easy to use functions.
    Note: For the time being, since the APIs are not ready the api client will call a direct db client instead of
    calling actual api end points. For this purpose, we designed two separate classes api_client and db_client.
    """

    def __init__(self, api_url=None, host=None, port=None, context_root=None, server=None, db=None, user=None,
                 password=None, client_type="db"):
        """
        Initializes the api client.
        :param api_url:
        :param host:
        :param port:
        :param context_root:
        """
        # Since we are using the db client for temporary purpose, we are not parameterising the arguments for the db
        # client but only parameterising the variables for api client.
        # We will instruct the function to read the parameters directly from the configuration file for db client.
        # We will use a hardcoded if condition variable and will switch back once the apis are ready
        if client_type == "db":
            self.client = DBClient(server=server, db=db, user=user, password=password)
        else:
            self.client = APIClient()

    def list_models(self, text=None, uuid=None, tag=None):
        """
        Lists all the models from the db. If text is given, then runs a full-text search on uuid and tag
        :param text:
        :param uuid:
        :param tag:
        :return:
        """
        return self.client.list_models(text, uuid, tag)