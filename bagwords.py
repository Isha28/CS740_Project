import pandas as pd
import csv

filename = "sample2.csv"

def process_csv(filename):
    example_file = open(filename, encoding="utf-8")
    example_reader = csv.reader(example_file)
    example_data = list(example_reader)
    example_file.close()
    return example_data

def main():
    with open("devicelist.txt") as f:
        target_macs = [line.rstrip().lower() for line in f]
        device_map = {}
        for mac in target_macs:
            device_mac = mac.split("-")[0]
            device_name = mac.split("-")[1]
            device_map[device_mac] = device_name
        
        print ("Device mappings", device_map)
        
        csv_rows = process_csv(filename)
        csv_head = csv_rows[0]
        csv_data = csv_rows[1:]
        
        #print ("CSV header", csv_head)

        bag_of_words = {}
        for row in csv_rows:
            for idx in range(len(csv_head)):
                if csv_head[idx] == "dest_mac" and row[idx] in device_map:
                    device_name = device_map[row[idx]]
                    
                    if device_name not in bag_of_words:     
                        bag_of_words[device_name] = set()

                    source_port = row[csv_head.index("source_port")]
                    dest_port = row[csv_head.index("dest_port")]
                    domain = row[csv_head.index("domain")]
                    
                    if source_port != "":
                        bag_of_words[device_name].add(source_port)

                    if dest_port != "":
                        bag_of_words[device_name].add(dest_port)

                    if domain != "": 
                        bag_of_words[device_name].add(domain)

        print ('Bag of words', bag_of_words)

if __name__ == "__main__":
    main()