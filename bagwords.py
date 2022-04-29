import pandas as pd
import csv

def process_csv(filename):
    example_file = open(filename, encoding="utf-8")
    example_reader = csv.reader(example_file)
    example_data = list(example_reader)
    example_file.close()
    return example_data

def generate_bag_of_words(filename):
    with open("devicelist.txt") as f:
        target_macs = [line.rstrip().lower() for line in f]
        device_map = {}
        for mac in target_macs:
            device_mac = mac.split("-")[0]
            device_name = mac.split("-")[1]
            device_map[device_mac] = device_name
        
        # print ("Device mappings", device_map)
        
        csv_rows = process_csv(filename)
        csv_head = csv_rows[0]
        csv_data = csv_rows[1:]
        
        #print ("CSV header", csv_head)

        rem_ports_keywords = {}
        domain_keywords = {}
        
        for row in csv_rows:
            for idx in range(len(csv_head)):
                if csv_head[idx] == "dest_mac" and row[idx] in device_map:
                    device_name = device_map[row[idx]]
                    rem_port = row[csv_head.index("dest_port")]
                    domain = row[csv_head.index("domain")]
                    
                    if rem_port != "":
                        if device_name not in rem_ports_keywords:     
                            rem_ports_keywords[device_name] = set()              
                        rem_ports_keywords[device_name].add(rem_port)
         
                    if domain != "": 
                        if device_name not in domain_keywords:     
                            domain_keywords[device_name] = set()
                        domain_keywords[device_name].add(domain)

        # print ('Bag of words', keywords)
        rem_ports_wordset = []
        for device in rem_ports_keywords:
            rem_ports_wordset.extend(list(rem_ports_keywords[device]))
        rem_ports_wordset = list(set(rem_ports_wordset))
        
        domain_wordset = []
        for device in domain_keywords:
            domain_wordset.extend(list(domain_keywords[device]))
        domain_wordset = list(set(domain_wordset))
        
        rem_ports_bag_of_words = {}
        for device in rem_ports_keywords:
            if device not in rem_ports_bag_of_words:
                rem_ports_bag_of_words[device] = {}
            for word in rem_ports_wordset:
                if word not in rem_ports_keywords[device]:
                    rem_ports_bag_of_words[device][word] = 0
                else:
                    rem_ports_bag_of_words[device][word] = 1
                    
        domain_bag_of_words = {}
        for device in domain_keywords:
            if device not in domain_bag_of_words:
                domain_bag_of_words[device] = {}
            for word in domain_wordset:
                if word not in domain_keywords[device]:
                    domain_bag_of_words[device][word] = 0
                else:
                    domain_bag_of_words[device][word] = 1
                    
        rem_ports_bag_of_words_df = pd.DataFrame.from_dict(rem_ports_bag_of_words, orient='index')
        domain_bag_of_words_df = pd.DataFrame.from_dict(domain_bag_of_words, orient='index')
        
        return (rem_ports_bag_of_words_df, domain_bag_of_words_df)

def main():
    pass

if __name__ == "__main__":
    main()