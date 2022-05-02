import pandas as pd
import csv

def process_csv(filename):
    example_file = open(filename, encoding="utf-8")
    example_reader = csv.reader(example_file)
    example_data = list(example_reader)
    example_file.close()
    return example_data

def fill_domain(filename):
    csv_rows = process_csv(filename)
    csv_head = csv_rows[0]
    csv_data = csv_rows[1:]
    dest_ip_domain = {}
    for row in csv_data:
        dest_ip = row[csv_head.index("dest_ip")]
        domain = row[csv_head.index("domain")]
        if domain != "" and domain[0].isalpha():
            dest_ip_domain[dest_ip] = domain

    print ("dest ip - domain", dest_ip_domain)
    
    df = pd.read_csv(filename)
    #df = df.reset_index()  # make sure indexes pair with number of rows
    
    #OPTIONAL : converted mdns to dns domain names too
    for index, row in df.iterrows():   
        if df.loc[index, 'dest_ip'] in dest_ip_domain:
            if str(df.loc[index, 'domain']) == "nan" or str(df.loc[index, 'domain'])[0].isalpha() == False:
                df.loc[index, 'domain'] = dest_ip_domain[df.loc[index, 'dest_ip']]
                #print(index, row['dest_ip'], row['domain'], df.loc[index, 'dest_ip'], df.loc[index, 'domain']) 
    print("Fill domain saves " + str(len(df)) + " rows")
    df.to_csv(filename, index=False)
    
def bucketize_flows(device_flow_map):
    flow_dict = {}
    for device in device_flow_map:
        for flow_id in device_flow_map[device]:
            flow_dict[flow_id] = device
    return flow_dict


def generate_bag_of_words(filename, attribute=None):
    fill_domain(filename)

    with open("devicelist.txt") as f:
        target_macs = [line.rstrip().lower() for line in f]
        device_map = {}
        for mac in target_macs:
            device_mac = mac.split("-")[0]
            device_name = mac.split("-")[1]
            device_map[device_mac] = device_name
        
        # print ("Device mappings", device_map)
        
        csv_rows = process_csv(filename)
        print("gen_bag_words reads " + str(len(csv_rows)) + " of data")
        csv_head = csv_rows[0]
        csv_data = csv_rows[1:]
        
#         print ("CSV header", csv_head)

        all_flow_ids = list() 
        device_flow_id_map = {}
        for row in csv_rows:
            flow_id = row[0]
            all_flow_ids.append(flow_id)
            dest_mac = row[csv_head.index("dest_mac")]
            src_mac = row[csv_head.index("source_mac")]
            if src_mac in device_map:
                device_name = device_map[src_mac]
            elif dest_mac in device_map:
                device_name = device_map[dest_mac]
            else:
                print("Device not found!")
                continue
            if device_name not in device_flow_id_map:
                device_flow_id_map[device_name] = set()
            device_flow_id_map[device_name].add(flow_id)
            
#         print ("device_flow_id_map ", device_flow_id_map)
        
        keywords = {}
        
        for row in csv_rows:
            flow_id = row[0]
            if (attribute == "ports"):
                field_value = row[csv_head.index("dest_port")]
            elif (attribute == "domains"):
                field_value = row[csv_head.index("domain")]
            elif (attribute == "cipher_suites"):
                field_value = row[csv_head.index("cipher_suites")]

            if (attribute == "ports" or attribute == "domains"):
                if field_value != "":
                    if flow_id not in keywords:     
                        keywords[flow_id] = set()              
                    keywords[flow_id].add(field_value)
            elif (attribute == "cipher_suites"):
                if field_value != "":
                    if flow_id not in keywords:
                        keywords[flow_id] = list()
                    keywords[flow_id].extend(field_value.split('|'))
            

        wordset = []
        for flow_id in keywords:
            wordset.extend(list(keywords[flow_id]))
        wordset = list(set(wordset))


        bag_of_words = {}
        for flow_id in all_flow_ids:
            if flow_id not in bag_of_words:
                bag_of_words[flow_id] = {}
            for word in wordset:
                if flow_id not in keywords or word not in keywords[flow_id]:
                    bag_of_words[flow_id][word] = 0
                else:
                    bag_of_words[flow_id][word] = 1

        bag_of_words_df = pd.DataFrame.from_dict(bag_of_words, orient='index')
        bag_of_words_df.drop(bag_of_words_df.index[0],inplace=True)

        return (bucketize_flows(device_flow_id_map), bag_of_words_df)

def main():
    #fill_domain("sample2.csv")
    #print(generate_bag_of_words("sample2.csv"))
    pass

if __name__ == "__main__":
    main()
