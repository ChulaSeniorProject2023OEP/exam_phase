# Convert .txt files to pandas DataFrame

import pandas as pd
import os
from tqdm import tqdm

def convert_txt_to_df(input_dir):
    '''
    Converts .txt files to pandas DataFrame and saves as .csv files.
    
    Args:
        input_dir (str): Directory containing input .txt files.
        output_dir (str): Directory to save output .csv files.
    '''
    for root, dirs, files in os.walk(input_dir):
        for filename in tqdm(files):
            if filename.endswith('.txt'):
                input_file_path = os.path.join(root, filename)
                output_file_path = os.path.join(root, filename.replace('.txt', '.csv'))
                
                with open(input_file_path, 'r') as f:
                    lines = f.readlines()
                    data = [line.strip().split() for line in lines]
                    print(data)
                    df = pd.DataFrame(data, columns=['start_time', 'end_time', 'label'])
                    
                # df = pd.read_csv(input_file_path, sep="\t", header=None, names=['start_time', 'end_time', 'label'])
                df.to_csv(output_file_path, index=False)
                
                print(f'Converted {input_file_path} to {output_file_path}')
                
if __name__ == '__main__':
    input_dir = '/Users/sarmkunatham/Desktop/1. Projects/GithubCodes/CU_Work/senior_project/exam_phase/data/raw/OEP_database'
    convert_txt_to_df(input_dir)

