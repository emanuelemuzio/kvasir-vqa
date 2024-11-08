import sys
sys.path.append('src')

import warnings
warnings.filterwarnings('ignore')

from classifier.data import retrieve_kvasir_vqa_dataset

if __name__ == '__main__':
    retrieve_kvasir_vqa_dataset()