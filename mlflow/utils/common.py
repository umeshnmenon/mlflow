import os
#import pwd
import getpass

# def get_username():
#     """
#     Gets the logged in user name
#     :return:
#     """
#     return pwd.getpwuid(os.getuid())[0]


def get_username2():
    """
    Gets the logged in user name
    :return:
    """
    return getpass.getuser()


def get_username3():
    """
    Gets the logged in user name
    :return:
    """
    return os.getlogin()