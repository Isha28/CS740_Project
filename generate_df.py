# coding=utf8

import pandas as pd
import numpy as np
import flow_stats
import time

filename = "april_lab.pcap"
t1 = time.time()
df = pd.DataFrame.from_dict(flow_stats.flow_statistics(filename), orient='index')
t2 = time.time()
df.to_csv(str.split(filename,".")[0]+ ".csv")
print("Parsing and storing pcap df took " + str(t2-t1) + " seconds")