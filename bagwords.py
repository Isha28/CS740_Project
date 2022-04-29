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

        keywords = {}
        for row in csv_rows:
            for idx in range(len(csv_head)):
                if csv_head[idx] == "dest_mac" and row[idx] in device_map:
                    device_name = device_map[row[idx]]
                    
                    if device_name not in keywords:     
                        keywords[device_name] = set()

                    source_port = row[csv_head.index("source_port")]
                    dest_port = row[csv_head.index("dest_port")]
                    domain = row[csv_head.index("domain")]
                    
                    if source_port != "":
                        keywords[device_name].add(source_port)

                    if dest_port != "":
                        keywords[device_name].add(dest_port)

                    if domain != "": 
                        keywords[device_name].add(domain)

        # print ('Bag of words', keywords)
        wordset = []
        for device in keywords:
            wordset.extend(list(keywords[device]))
        wordset = list(set(wordset))
        # keywords = {key:list(value) for key, value in keywords.items()}
        bag_of_words = {}
        for device in keywords:
            if device not in bag_of_words:
                bag_of_words[device] = {}
            for word in wordset:
                if word not in keywords[device]:
                    bag_of_words[device][word] = 0
                else:
                    bag_of_words[device][word] = 1
        bag_of_words_df = pd.DataFrame.from_dict(bag_of_words, orient='index')
        return bag_of_words_df

def main():
    pass

if __name__ == "__main__":
    main()