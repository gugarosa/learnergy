import pickle

import torch

import recogners.utils.logging as l

logger = l.get_logger(__name__)


class Model:
    """The Model class is the basis for any custom model.

    One can configure, if necessary, different properties or methods that
    can be used throughout all childs.

    """

    def __init__(self):
        """Initialization method.

        """

        # Setting default tensor type to Double
        torch.set_default_tensor_type(torch.DoubleTensor)

        # Creating an empty dictionary to hold historical values
        self._history = {}

    @property
    def history(self):
        """dict: Dictionary containing historical values.

        """

        return self._history

    @history.setter
    def history(self, history):
        self._history = history

    def dump(self, **kwargs):
        """Dumps any amount of keyword documents to lists in the history property.

        """

        # Iterate through key-word arguments
        for k, v in kwargs.items():
            # Check if there is already an instance of current
            if k not in self.history.keys():
                # If not, creates an empty list
                self.history[k] = []

            # Appends the new value to the list
            self.history[k].append(v)

    def save(self, file_name):
        """Saves the object to a pickle encoding.

        Args:
            file_name (str): String holding the file's name that will be saved.

        """

        logger.info(f'Saving model: {file_name}.')

        # Opening the file in write mode
        f = open(file_name, 'wb')

        # Dumps to a pickle file
        pickle.dump(self, f)

        # Close the file
        f.close()

        logger.info('Model saved.')

    def load(self, file_name):
        """Loads the object from a pickle encoding.

        Args:
            file_name (str): String containing pickle's file path.

        """

        logger.info(f'Loading model: {file_name}.')

        # Opens the desired file in read mode
        f = open(file_name, 'rb')

        # Loads using pickle
        model = pickle.load(f)

        # Resetting current object state to loaded state
        self.__dict__.update(model.__dict__)

        logger.info('Model loaded.')
