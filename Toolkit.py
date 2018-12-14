__author__ = 'timo'
"""
Usefull Code Snippets.

"""


import sys


import numpy as np  # for file
import logging      # for file
import os           # for file
from datetime import datetime # for file


class Tk_console(object):
    @staticmethod
    def progressBar(iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 50):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
        """
        # Credits: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console

        filledLength    = int(round(barLength * iteration / float(total)))
        percents        = round(100.00 * (iteration / float(total)), decimals)
        bar             = '#' * filledLength + '-' * (barLength - filledLength)

        # not working with Pycharm console, but better in shell (logging in new line)
        sys.stdout.write('%s [%s] %s%s %s \r' % (prefix, bar, percents, '%', suffix))
        # working in Pycharm
        #sys.stdout.write('\r%s [%s] %s%s %s' % (prefix, bar, percents, '%', suffix))

        sys.stdout.flush()
        if iteration == total:
            print("\n")

class Tk_file(object):
    @staticmethod
    def saveToFile(aArrayLike, aFilePath, mode='npy', overwrite=False):
        """
        Saves array-like sturctures and object to files or images.
        :param aArrayLike:
        :param aFilePath:  for saving to subdirectory of current directory: ./subdir/filename
        :param mode:
        :param overwrite:
        :return: name of filePath
        """

        if aFilePath == "":
            logging.warn("Nothing saved, no filePath given.")
            return

        # add file extension
        if not '.' + mode in aFilePath:
            aFilePath = aFilePath + '.' + mode

        # get new, unused filePath
        if overwrite is False:
            try:
                aFilePath = Tk_file._checkPathAlreadyExist(aFilePath)
            except:
                logging.error("File %s already exists and generating new filename failed. Nothing saved.", aFilePath)
                return

        # save
        if mode == 'txt':
            np.savetxt(aFilePath, aArrayLike, delimiter="\t")
        elif mode == 'npy':
            np.save(aFilePath, aArrayLike)
        elif mode == 'png':
            #im = Image.fromarray(aArrayLike)
            #im.save(aFilePath)
            cv2.imwrite(aFilePath, aArrayLike)
        elif mode == 'json':
            import json
            with open(aFilePath, 'w') as datafile:
                json.dump(aArrayLike, datafile, default=Tk_file._obj2json, sort_keys=True, indent=4)
        else:
            logging.warning('Unknown mode (%s) for saving data. Nothing saved!', mode)
            return

        logging.info('Saved successfully to file: %s ', aFilePath)
        return aFilePath

    @staticmethod
    def _checkPathAlreadyExist(path):
        """
        Checks whether a given path is existing in the file system.
        :param path: path to file or folder
        :return: path with subsequent <_timestamp> if path is already existing
        """

        if not os.path.exists(path):
            return path

        return Tk_file._addTimestamp(path)

    @staticmethod
    def _addTimestamp(filePath):
        """
        Adds a timestamp to a given filePath
        :param filePath:
        :return: <filePath>_<hour.minute.second.milisecond>.<extension>
        """
        noExtension = os.path.splitext(filePath)
        timestamp = str(datetime.now().time()).replace(":", ".")    # replacing illegal char on windows machines

        return noExtension[0] + "_" + timestamp + noExtension[1]

    @staticmethod
    def _obj2json(obj):
        """
        Converts objects into json seriazables.
        :return: obj, if no .__dict__ available. Else: json serializable
        """

        # handle numpy 1D arrays first. Can cause problems if nested or higher dimension!
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # 'normal' in-built types have no .__dict__ attribute -> will raise
        # user-built primitive classes save their data in __dict__
        try:
            ret = obj.__dict__
        except AttributeError:
            raise TypeError("Unserializable object {} of type {}".format(obj,type(obj)))

        return ret

    @staticmethod
    def _loadArrayData(aFilePath, mode='npy'):

        if mode == 'npy':
            arr = np.load(aFilePath)
        elif mode == 'json':
            with open(aFilePath, 'r') as json_file:
                arr = json.load(json_file)
        else:
            raise AttributeError('Unknown file loading mode %s.', mode)

        if arr is None:
            logging.exception("Failed to load array data. Path not found:" + str(aFilePath))
            raise IOError("Failed to load array. Path not found:" + str(aFilePath))

        logging.debug('Array data file successfully loaded from: %s', aFilePath)
        return arr

class Tk_list(object):
    @staticmethod
    def all_elements_equal(list):
        # http://stackoverflow.com/q/3844948/
        return not list or list.count(list[0]) == len(list)

    @staticmethod
    def find_closest_idx(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx