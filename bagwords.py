import pandas as pd
import csv
import os

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

    # print ("dest ip - domain", dest_ip_domain)
    
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

def fill_ciphers(filename):
    csv_rows = process_csv(filename)
    csv_head = csv_rows[0]
    csv_data = csv_rows[1:]
    dest_ip_ciphers = {}
    for row in csv_data:
        dest_ip = row[csv_head.index("dest_ip")]
        ciphers = row[csv_head.index("cipher_suites")]
        if ciphers != "":
            dest_ip_ciphers[dest_ip] = ciphers

    # print ("dest ip - ciphers", dest_ip_ciphers)
    
    df = pd.read_csv(filename)
    #df = df.reset_index()  # make sure indexes pair with number of rows
    
    #OPTIONAL : converted mdns to dns domain names too
    for index, row in df.iterrows():   
        if df.loc[index, 'dest_ip'] in dest_ip_ciphers:
            if str(df.loc[index, 'cipher_suites']) == "nan":
                df.loc[index, 'cipher_suites'] = dest_ip_ciphers[df.loc[index, 'dest_ip']]
                #print(index, row['dest_ip'], row['domain'], df.loc[index, 'dest_ip'], df.loc[index, 'domain']) 
    print("Fill ciphers saves " + str(len(df)) + " rows")
    df.to_csv(filename, index=False)
    
def bucketize_flows(device_flow_map):
    flow_dict = {}
    for device in device_flow_map:
        for flow_id in device_flow_map[device]:
            flow_dict[flow_id] = device
    return flow_dict


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def generate_hourly_bag_of_words(filename,attribute=None):
    fill_domain(filename)
    fill_ciphers(filename)
    with open("devicelist.txt") as f:
        target_macs = [line.rstrip().lower() for line in f]
    device_map = {}
    for mac in target_macs:
        device_mac = mac.split("-")[0]
        device_name = mac.split("-")[1]
        device_map[device_mac] = device_name
    csv_rows = process_csv(filename)
    # print("gen_hourly_bag_words reads " + str(len(csv_rows)) + " of data")
    csv_head = csv_rows[0][1:]
    csv_head[0] = "flow_id"
    # print(csv_head)
    csv_data = [row[1:] for row in csv_rows[1:]]
    # print(csv_data[0])
    keywords = {}
    all_flow_ids = list() 
    all_src_macs = set()
    device_flow_id_map = {}
    for row in csv_data:
        flow_id = row[0]

        all_flow_ids.append(flow_id)
        dest_mac = row[csv_head.index("dest_mac")]
        src_mac = row[csv_head.index("source_mac")]
        all_src_macs.add(src_mac)
        if src_mac in device_map:
            device_name = device_map[src_mac]
        elif dest_mac in device_map:
            device_name = device_map[dest_mac]
        else:
            # print("Device not found!")
            continue
        if device_name not in device_flow_id_map:
            device_flow_id_map[device_name] = set()
        device_flow_id_map[device_name].add(flow_id)
        if (attribute == "ports"):
            field_value = row[csv_head.index("dest_port")]
        elif (attribute == "domains"):
            field_value = row[csv_head.index("domain")]
        elif (attribute == "cipher_suites"):
            field_value = row[csv_head.index("cipher_suites")]

        source_mac = src_mac

        if (attribute == "ports" or attribute == "domains"):
            if field_value != "":
                if source_mac not in keywords:     
                    keywords[source_mac] = {}
                if field_value not in keywords[source_mac]:
                    keywords[source_mac][field_value] = 0      
                keywords[source_mac][field_value] += 1 
        elif (attribute == "cipher_suites"):
            if field_value != "":
                if source_mac not in keywords:
                    keywords[source_mac] = list()
                keywords[source_mac].extend(field_value.split('|'))
    # print ("device_flow_id_map ", device_flow_id_map)
    # print("Keywords dictionary created")
       
    # print(keywords)   
        

    wordset = []
    if attribute == "ports" or attribute == 'domains':
        for source_mac in keywords:
            wordset.extend(list(keywords[source_mac].keys()))
    elif attribute == "cipher_suites":
        for source_mac in keywords:
            wordset.extend(list(keywords[source_mac]))
    wordset = list(set(wordset))
    # print(str(len(wordset)) + " words present in the wordset")
    # print(wordset)
    # print("Creating bag of words dictionary")
    hourly_bag_of_words = {}
    for mac in all_src_macs:
        if mac not in hourly_bag_of_words:
            hourly_bag_of_words[mac] = {}
        for word in wordset:
            if mac not in keywords or word not in keywords[mac]:
                hourly_bag_of_words[mac][word] = 0
            else:
                hourly_bag_of_words[mac][word] = keywords[mac][word]
    # print("Creating dataframe from dictionary")
    hourly_bag_of_words_df = pd.DataFrame.from_dict(hourly_bag_of_words, orient="index")
    hourly_bag_of_words_df = hourly_bag_of_words_df[hourly_bag_of_words_df.index != "source_mac"]
    
    hourly_bag_of_words_df.reset_index(inplace=True)
   
    hourly_bag_of_words_df = hourly_bag_of_words_df.rename(columns = {'index':'class'})
    # print(device_map)
    hourly_bag_of_words_df = hourly_bag_of_words_df[hourly_bag_of_words_df["class"].isin(device_map)]
    hourly_bag_of_words_df["class"] = hourly_bag_of_words_df["class"].apply(lambda x: device_map[x])
    # print(hourly_bag_of_words_df)
    # print("Created bag of words dataframe. Returning")
    device_flow_id_map = {device:list(flow_ids) for (device, flow_ids) in device_flow_id_map.items()}
    # print(device_flow_id_map)
    hourly_bag_of_words_df = hourly_bag_of_words_df.astype(str)

    return (device_flow_id_map, hourly_bag_of_words_df)
    
    # for flow_id in all_flow_ids:
    #     if flow_id not in bag_of_words:
    #         bag_of_words[flow_id] = {}
    #     for word in wordset:
    #         if flow_id not in keywords or word not in keywords[flow_id]:
    #             bag_of_words[flow_id][word] = 0
    #         else:
    #             bag_of_words[flow_id][word] = 1
    # print("Creating dataframe from dictionary")
    # bag_of_words_df = pd.DataFrame.from_dict(bag_of_words, orient='index')
    # bag_of_words_df.drop(bag_of_words_df.index[0],inplace=True)
    # print("Created bag of words dataframe. Returning")
    # return (bucketize_flows(device_flow_id_map), bag_of_words_df)



def generate_bag_of_words(filename, attribute=None):
    print("Bag words function begins")
    # fill_domain(filename)
    # fill_ciphers(filename)
    with open("devicelist.txt") as f:
        target_macs = [line.rstrip().lower() for line in f]
    device_map = {}
    for mac in target_macs:
        device_mac = mac.split("-")[0]
        device_name = mac.split("-")[1]
        device_map[device_mac] = device_name
    
        print ("Device mappings", device_map)
        
    csv_rows = process_csv(filename)
    print("gen_bag_words reads " + str(len(csv_rows)) + " of data")
    csv_head = csv_rows[0]
    csv_data = csv_rows[1:]
    
    print ("CSV header", csv_head)
    keywords = {}
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
    print ("device_flow_id_map ", device_flow_id_map)
    print("Keywords dictionary created")
       
    print(keywords)
        

    wordset = []
    for flow_id in keywords:
        wordset.extend(list(keywords[flow_id]))
    wordset = list(set(wordset))
    print(str(len(wordset)) + " words present in the wordset")
    print("Creating bag of words dictionary")
    bag_of_words = {}

    for flow_id in all_flow_ids:
        if flow_id not in bag_of_words:
            bag_of_words[flow_id] = {}
        for word in wordset:
            if flow_id not in keywords or word not in keywords[flow_id]:
                bag_of_words[flow_id][word] = 0
            else:
                bag_of_words[flow_id][word] = 1
    print("Creating dataframe from dictionary")
    bag_of_words_df = pd.DataFrame.from_dict(bag_of_words, orient='index')
    bag_of_words_df.drop(bag_of_words_df.index[0],inplace=True)
    print("Created bag of words dataframe. Returning")

    bag_of_words_df = bag_of_words_df.astype(str)

    return (bucketize_flows(device_flow_id_map), bag_of_words_df)

def merge_hourly_instances(dirname,attribute):
    for idx, path in enumerate(os.listdir(dirname)):
        if path[-3:] == 'csv':
            if idx == 0:
                mapping, merged_df = generate_hourly_bag_of_words(dirname + path,attribute=attribute)
                merged_df["mapping"] = merged_df["class"].apply(lambda x: '|'.join(mapping[x]))
            else:
                mapping, new_df = generate_hourly_bag_of_words(dirname + path,attribute=attribute)
                new_df["mapping"] = new_df["class"].apply(lambda x: '|'.join(mapping[x]))
                merged_df = merged_df.append(new_df,ignore_index = True)
    merged_df.to_csv("merged_bag_of_words_cipher_suites.csv",index=False)
    print(merged_df)
    return merged_df


def main():
    merge_hourly_instances("hourly_output/","cipher_suites")
    # generate_hourly_bag_of_words("hourly_output/hour_0.csv",attribute="ports")
    #fill_domain("sample2.csv")
    #print(generate_bag_of_words("sample2.csv"))
    pass

if __name__ == "__main__":
    main()
