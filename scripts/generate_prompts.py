import sys
sys.path.append('src')

import warnings
warnings.filterwarnings('ignore')

from classifier.data import generate_prompt_dataset
from common.util import logger

if __name__ == '__main__':
    logger.info("Launching generate_prompts script")
    generate_prompt_dataset()