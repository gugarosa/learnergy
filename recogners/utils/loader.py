import pandas as pd

import recogners.utils.logging as l

logger = l.get_logger(__name__)


def load_opf(path):
    """Loads a file in OPF format.

    Note that the input file has to be a text file.

    Args:
        path (str): A string containing the path to the OPF .txt file.

    Returns:
        A (X, Y) pair holding the file's samples.

    """

    logger.debug(f'Loading OPF text file: {path}.')

    # Tries to read .txt file into a dataframe
    try:
        # Actually reads the file
        opf = pd.read_csv(path, sep=' ', skiprows=1, header=None)

    # If file is not found,
    except FileNotFoundError as e:
        # Handle the exception and exit
        logger.error(e)
        raise

    # Gathering the labels
    Y = opf.iloc[:, 1].values

    # Gathering the features
    X = opf.iloc[:, 2:].values

    logger.debug('File loaded.')

    return X, Y
