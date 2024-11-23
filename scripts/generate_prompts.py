import sys
sys.path.append('src')

import warnings
warnings.filterwarnings('ignore')

from classifier.data import generate_prompt_dataset

if __name__ == '__main__':
    generate_prompt_dataset()