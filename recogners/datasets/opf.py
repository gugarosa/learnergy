import recogners.utils.loader as loader
import recogners.utils.logging as l
from recogners.core.dataset import Dataset

logger = l.get_logger(__name__)


class OPFDataset(Dataset):
    """An OPF dataset class, inherited from Recogners' dataset.

    """

    def __init__(self, path=None):
        """Initialization method.

        Args:
            path (str): A string containing the path to the OPF .txt file.

        """

        logger.info('Overriding class: Dataset -> OPFDataset.')

        # Loading OPF .txt file
        X, Y = loader.load_opf(path)

        # Override its parent class with the proper parameters
        super(OPFDataset, self).__init__(X, Y)

        logger.info('Class overrided.')
