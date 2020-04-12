import string
from connectors.sql_client import SQLClient
from connectors.snowflake_client import SnowflakeClient

class DataClient:
    """
    The class establishes a connection between the python app and the database. Currently supports MS SQL Server and
    Snowflake
    """

    def __init__(self, provider="SQL", server=None, db=None, user=None, password=None, connection_string=None):
        self.server = server
        self.db = db
        self.user = user
        self.password = password
        self.connection_string = connection_string
        if provider == "SQL":
            self.setup_sql_connection()
        elif string.lower(provider) == "snowflake":
            self.setup_snowflake_connection()


    def setup_sql_connection(self):
        self.cursor = SQLClient(self.server, self.db, self.user, self.password, self.connection_string).cursor

    def setup_snowflake_connection(self):
        self.cursor = SnowflakeClient(self.server, self.db, self.user, self.password, self.connection_string).cursor