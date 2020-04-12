import sqlalchemy
import pyodbc
import urllib


class SQLClient:
    """
    Establishes a connection to MS SQL Server
    """

    def __init__(self, server=None, db=None, user=None, password=None, connection_string=None):

        # default odbc format:
        odbc_string = 'DRIVER={driver};SERVER={server};DATABASE={db};UID={user_id};PWD={password};ansi=True'

        # Some checks. Either connection string or details must be given
        if connection_string is None:
            if server is None:
                assert False, "Server name must be provided"
            if db is None:
                assert False, "DB name must be provided"
            if user is None:
                assert False, "User id must be provided"
            if password is None:
                assert False, "Password must be provided"
            connection_string = odbc_string.format(
                # driver='{SQL Server}',
                driver='{ODBC Driver 17 for SQL Server}',
                server=server,
                db=db,
                user_id=user,
                password=password)

        cnxn = pyodbc.connect(connection_string)
        cursor = cnxn.cursor()
        self.cursor = cursor

        self._engine = self.get_engine(server, db, user, password, connection_string)
        #self._engine = sqlalchemy.create_engine(
        #    "mssql+pyodbc:///?odbc_connect=%s" % urllib.parse.quote_plus(connection_string))

    @property
    def engine(self):
        return self._engine

    def get_engine(self, server=None, db=None, user=None, password=None, connection_string=None):
        """
        Gets sqlalchemy engine
        :param password:
        :return:
        """
        # default odbc format:
        odbc_string = 'DRIVER={driver};SERVER={server};DATABASE={db};UID={user_id};PWD={password};ansi=True'

        # Some checks. Either connection string or details must be given
        if connection_string is None:
            if server is None:
                assert False, "Server name must be provided"
            if db is None:
                assert False, "DB name must be provided"
            if user is None:
                assert False, "User id must be provided"
            if password is None:
                assert False, "Password must be provided"

            connection_string = odbc_string.format(
                driver='{SQL Server}',
                server=server,
                db=db,
                user_id=user,
                password=password)

        self._engine = sqlalchemy.create_engine(
            "mssql+pyodbc:///?odbc_connect=%s" % urllib.parse.quote_plus(connection_string))

        return self._engine

