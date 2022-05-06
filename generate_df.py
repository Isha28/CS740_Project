# coding=utf8

import pandas as pd
import numpy as np
import flow_stats
import time
import argparse
import os

def args_parser():
    parser = argparse.ArgumentParser(description="Specify pcap file to parse")
    parser.add_argument("-f","--file",dest="filename",help="pcap filename")
    parser.add_argument("-d","--directory",dest="dirname",help="pcap dirname")
    args = parser.parse_args()
    return args

def process_file(filename):
    input_path = os.path.abspath(filename)
    output_file = filename+ ".csv"
    print(output_file)
    if os.path.exists(output_file):
        return
    t1 = time.time()
    df = pd.DataFrame.from_dict(flow_stats.flow_statistics(filename), orient='index')
    t2 = time.time()
    df.to_csv(output_file)
    print("Parsing and storing pcap df took " + str(t2-t1) + " seconds")

def process_dir(dirname):
    for path in os.listdir(dirname):
        if (path[-4:] == 'pcap'):
            print(path)
            process_file(dirname + path)

def main():
    args = args_parser()
    if(args.filename):
        filename = args.filename
        process_file(filename)
    elif(args.dirname):
        dirname = args.dirname
        process_dir(dirname)
        
if __name__ == "__main__":
    main()
