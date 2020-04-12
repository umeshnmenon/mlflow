"""
This file contains common functions across the MLFlow. Stuffs like reading the configuration dictionary and setting up
connectors etc.
"""
import tempfile, os, zipfile
from constants import *
from connectors.sql_client import SQLClient
from helpers.config_helper import ConfigHelper
from api.mlflow_api_client import MLFlowAPIClient

def create_mlflow_client():
    """
    A helper function to create connection to mlflow database tables
    :return:
    """
    config = ConfigHelper().get_config()
    server = config.get(MLFLOW, "server")
    db = config.get(MLFLOW, "db")
    user = config.get(MLFLOW, "user")
    password = os.getenv(MLFLOW_PWD) or (config.get(MLFLOW, "password") if config.has_option(MLFLOW, "password") else
                                         None)
    if password is None or password == '':
        assert False, "Cannot connect to mlflow database. Please provide password."
    #return SQLClient(server=server, db=db, user=user, password=password)
    return MLFlowAPIClient(server=server, db=db, user=user, password=password)

def create_data_client(config):
    """
    Creates a sql client based on the info given in the config dictionary
    :param config:
    :return:
    """
    server = config["server"]
    db = config["db"]
    user = config["user"]
    password = config["password"]
    if password is None or password == '':
        assert False, "Cannot connect to database. Please provide password."
    return SQLClient(server=server, db=db, user=user, password=password)


def zip_folder(folder, zip_name):
    """
    Zips the contents of a folder with a name
    :param folder:
    :param zip_name:
    :return:
    """
    ziph = zipfile.ZipFile(folder + '//' + zip_name + '.zip', 'w', zipfile.ZIP_DEFLATED)
    # ziph is zipfile handle
    for root, dirs, files in os.walk(folder):
        for file in files:
            ziph.write(os.path.join(root, file))

    ziph.close()


def convert_to_binary(filename):
    # Convert data to binary format
    with open(filename, 'rb') as file:
        binary_data = file.read()
    return binary_data