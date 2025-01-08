import sys
sys.path.append('src')

import pandas as pd
import json
import os
from dotenv import load_dotenv
from common.util import ROOT

load_dotenv()

if __name__ == '__main__':
    vocabulary = []
    metadata = pd.read_csv(os.getenv('KVASIR_VQA_CSV'))
    for answer in metadata['answer'].value_counts().keys():
        vocabulary.extend([answ.strip() for answ in answer.split(';')])
    vocabulary = list(set(vocabulary))
    with open(f"{ROOT}/{os.getenv('KVASIR_VQA_VOCABULARY')}", 'w', encoding='utf-8') as f:
        json.dump(vocabulary, f, ensure_ascii=False, indent=4)