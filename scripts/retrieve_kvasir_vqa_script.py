import sys
sys.path.append('src')

import warnings
warnings.filterwarnings('ignore')

from kvasirvqa.data import retrieve_kvasir_vqa_dataset
from common.util import logger

if __name__ == '__main__':
    logger.info("Retrieving KvasirVQA dataset files")
    retrieve_kvasir_vqa_dataset()