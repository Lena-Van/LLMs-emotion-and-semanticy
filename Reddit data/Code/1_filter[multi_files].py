import os
import re
from datetime import datetime
import pandas as pd
from multiprocessing import Pool

# keywords
keywords = ["climate change", "climate science", "climate manipulation", "climate Engineering", "Climate Hacking", 
            "climate modification", "climate science", "Global Warming", "carbon footprint", "The Paris Agreement"]

# file path
source_path = "/mnt/data/reddit/reddit/RC/"
target_path = "data/csv_filtered/"

# set time range
start_date = datetime.strptime("2014-01", "%Y-%m")
end_date = datetime.strptime("2022-12", "%Y-%m")

# settings for result filename
label = 'en+climatechange'
content = 'comment'

# format of source filename
date_str_pattern = r'RC_(.*?)\.zst\.csv'

# other settings
chunksize = 1e6
proc_num = 13

def process_file(source_name):
    # check file name
    mtch = re.search(date_str_pattern, source_name)
    if mtch:
        # get date from file name
        date_str = mtch.group(1)
        file_date = datetime.strptime(date_str, '%Y-%m')
        
        # check if the date in time range. if yes, process the file.
        if start_date <= file_date <= end_date:
            print(f"processing {source_name} ... ...\n", end='')
            target_name = f"{content}[{label}][{date_str}].csv"

            # check if the target file already exists
            if os.path.exists(target_path + target_name):
                print(f"{target_name} already exists.\n", end='')
                return

            try:
                for chunk in pd.read_csv(source_path + source_name, chunksize=chunksize, engine='python'):
                    df = chunk[chunk['body'].astype(str).str.contains('|'.join(keywords), case=False)]
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
