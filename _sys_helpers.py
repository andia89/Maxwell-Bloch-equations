import shutil
import jupyter_client
import filecmp
import pickle


def get_connection_file():
    return jupyter_client.find_connection_file()

def copy_connection_file():
    f = get_connection_file()
    new_f = "/home/andreas/Dropbox/connection_files/current_connection.json"
    if not filecmp.cmp(new_f, f):
        shutil.copyfile(f, new_f)

def load(filename):
    f = open(filename, "rb")
    ret = pickle.load(f)
    f.close()
    return ret