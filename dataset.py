
import os 
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()

'''
Download the KVASIR-VQA dataset, splitted in the metadata.csv file and the imgs
'''

def load() -> any:
    return load_dataset(os.getenv('KVASIR_VQA_DATASET'))

def retrieve_dataset() -> None:
    dataset = load()
    dataframe = dataset['raw'].select_columns(['source', 'question', 'answer', 'img_id']).to_pandas()
    
    if not os.path.exists(f"{os.getenv('KVASIR_VQA_METADATA')}/metadata.csv"):
        dataframe.to_csv(f"{os.getenv('KVASIR_VQA_METADATA')}/metadata.csv", index=False)
        
    if not os.path.exists(f"{os.getenv('KVASIR_VQA_DATA')}"):
        os.makedirs(f"{os.getenv('KVASIR_VQA_DATA')}", exist_ok=True)
        for i, row in dataframe.groupby('img_id').nth(0).iterrows(): # for images
            dataset['raw'][i]['image'].save(f"{os.getenv('KVASIR_VQA_DATA')}/{row['img_id']}.jpg")
            
def main():
    retrieve_dataset()
    
if __name__ == '__main__':
    main()