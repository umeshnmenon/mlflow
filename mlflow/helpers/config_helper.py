import configparser, sys, argparse, os, time
from autologging import logged
import logging
from utils.files import get_full_path

@logged(logging.getLogger())
class ConfigHelper:
    """
    A simple config loader class to load the configuration file
    """

    def __init__(self, config_file = None, env = None):
        self.__log.info("Initializing Config")
        self.__log.info("Reading config file")
        if config_file is None:
            config_file = self._get_config_file(env)
        self.config = self._get_config(config_file)
        self.env = env

    def _get_config_file(self, env = None):
        """
        Reads the configuration file
        :param env:
        :return:
        """
        config_file = "settings.ini"
        root_folder = get_full_path()# os.path.dirname(os.path.realpath(__file__))
        if env is None:
            env = os.getenv("env", "")

        if env:
            config_file = "settings_{}.ini"
            config_file = config_file.format(env)

        self.config_file = root_folder + '/' + config_file
        # convert to raw string to take care of backslashes
        #self.config_file = repr(self.config_file)
        #assert os.path.exists(self.config_file)
        self.__log.info("Config file to use: {}".format(self.config_file))
        return self.config_file

    def _get_config(self, config_file):
        """
        Loads the configuration from the configuration file
        :param config_file:
        :return:
        """
        lconfig = configparser.ConfigParser()
        lconfig.read(config_file)
        return lconfig


    def get_config(self):
        """
        Just returns the config
        :return:
        """
        return self.config