# coding=utf8

import pandas as pd
import numpy as np
import flow_stats
import time
import argparse

def args_parser():
    parser = argparse.ArgumentParser(description="Specify pcap file to parse")
    parser.add_argument("-f","--file",dest="filename",help="pcap filename")
    args = parser.parse_args()
    return args

def main():
    args = args_parser()
    filename = args.filename
    t1 = time.time()
    df = pd.DataFrame.from_dict(flow_stats.flow_statistics(filename), orient='index')
    t2 = time.time()
    df.to_csv(str.split(filename,".")[0]+ ".csv")
    print("Parsing and storing pcap df took " + str(t2-t1) + " seconds")

if __name__ == "__main__":
    main()
