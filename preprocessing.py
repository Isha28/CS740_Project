import os
import pandas as pd
import numpy as np
import flow_stats
import time
import argparse
import os
import csv

def process_csv(filename):
    example_file = open(filename, encoding="utf-8")
    example_reader = csv.reader(example_file)
    example_data = list(example_reader)
    example_file.close()
    return example_data

def merge_files(dirname, output_path):
    files = []
    for path in os.listdir(dirname):
        if (os.path.isfile(path) and path.split('.')[-1] == "csv"):
            files.append(path)
    merged_rows = []
    t_count = 1
    u_count = 1
    for idx, file in enumerate(files):
        rows = process_csv(file)
        for row in rows[1:]:
            if row[0][0] == 't':
                row[0] = 't' + str(t_count)
                t_count +=1 
            elif row[0][0] == 'u':
                row[0] = 'u' + str(u_count)
                u_count +=1 
        if idx == 0:
            merged_rows.extend(rows)
        else:
            merged_rows.extend(rows[1:])
    df = pd.DataFrame(merged_rows)
    df.columns = df.iloc[0]
    df.drop(df.index[0],inplace=True)
    df.set_index(df.iloc[:, 0], inplace=True)
    df.to_csv(output_path,index=False)

def args_parser():
    parser = argparse.ArgumentParser(description="Specify CSV files to merge")
    parser.add_argument("-d","--directory",dest="dirname",help="directory of csv files")
    args = parser.parse_args()
    return args

def main():
    args = args_parser()
    dirname = args.dirname
    merge_files(dirname, "merged_traces_sample.csv")

if __name__ == "__main__":
    main()
