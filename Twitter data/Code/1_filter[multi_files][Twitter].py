import os
import re
import ast
from datetime import datetime
import pandas as pd
from multiprocessing import Pool
import argparse


parse = argparse.ArgumentParser()


parse.add_argument("--keywords")  
parse.add_argument("--file_number_range", default="(1, 1)")
parse.add_argument("--result_label", default='')
parse.add_argument("--proc_num", default=10)
parse.add_argument("--source_path", default="/mnt/data/Project7/fakenews/csv/")
parse.add_argument("--target_path", default="data/csv_filtered/")

args = parse.parse_args()


# keywords
keywords = ast.literal_eval(args.keywords)

# file path
source_path = args.source_path
target_path = args.target_path

# set file range
file_number_range = ast.literal_eval(args.file_number_range)
start_num = file_number_range[0]
end_num = file_number_range[1]

# settings for result filename
label = args.result_label  # 'en+POTUS2020'
content = 'tweet'

# format of source filename
file_num_pattern = r'(\d+)\.csv'

# other settings
chunksize = 1e6
proc_num = int(args.proc_num)

# print arguments
print("\nArguments are listed bellow.")
print(f"keywords: {keywords}")
print(f"start_file_num: {start_num}")
print(f"end_file_num: {end_num}")
print(f"result_label: {label}")
print(f"number of process: {proc_num}")
print(f"source_path: {source_path}")
print(f"target_path: {target_path}")
print()

def process_file(source_name):
    # check file name
    mtch = re.search(file_num_pattern, source_name)
    if mtch:
        # get file number from file name
        file_num = int(mtch.group(1))
        
        # check if the date in time range. if yes, process the file.
        if start_num <= file_num <= end_num:
            print(f"processing {source_name} ... ...\n", end='')
            target_name = f"{content}[{label}][{file_num}].csv"

            # check if the target file already exists
            if os.path.exists(target_path + target_name):
                print(f"{target_name} already exists.\n", end='')
                return

            try:
                for chunk in pd.read_csv(source_path + source_name, chunksize=chunksize, low_memory=False):
                    df = chunk[
                            chunk['text'].str.contains('|'.join(keywords), case=False) |
                            chunk['entities.hashtags'].astype(str).str.contains('|'.join(keywords), case=False)
                        ]
                    df.to_csv(target_path + target_name, mode='a', index=False)
                print(f"{source_name} processed.\n", end='')
            except Exception as e:
                print(f"failed processing {source_name}.\n{e}\n", end='')

# get a list of files
files = os.listdir(source_path)

# create a pool of workers
with Pool(processes=proc_num) as pool:
    pool.map(process_file, files)

print("Done.")
