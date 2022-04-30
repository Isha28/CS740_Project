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
    
    df.to_csv(filename, index=False)
    
def generate_bag_of_words(filename):
    fill_domain("sample2.csv")

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
        
#         print ("CSV header", csv_head)

        device_flow_id_map = {}
        for row in csv_rows:
            flow_id = row[0]
            dest_mac = row[csv_head.index("dest_mac")]
            if dest_mac not in device_map:
                continue
            device_name = device_map[dest_mac]
            if device_name not in device_flow_id_map:
                device_flow_id_map[device_name] = set()
            device_flow_id_map[device_name].add(flow_id)
            
#         print ("device_flow_id_map ", device_flow_id_map)
        
        rem_ports_keywords = {}
        domain_keywords = {}
        
        for row in csv_rows:
            flow_id = row[0]  
            rem_port = row[csv_head.index("dest_port")]
            domain = row[csv_head.index("domain")]

            if rem_port != "":
                if flow_id not in rem_ports_keywords:     
                    rem_ports_keywords[flow_id] = set()              
                rem_ports_keywords[flow_id].add(rem_port)

            if domain != "": 
                if flow_id not in domain_keywords:     
                    domain_keywords[flow_id] = set()
                domain_keywords[flow_id].add(domain)

        rem_ports_wordset = []
        for flow_id in rem_ports_keywords:
            rem_ports_wordset.extend(list(rem_ports_keywords[flow_id]))
        rem_ports_wordset = list(set(rem_ports_wordset))
        
        domain_wordset = []
        for flow_id in domain_keywords:
            domain_wordset.extend(list(domain_keywords[flow_id]))
        domain_wordset = list(set(domain_wordset))
        
        rem_ports_bag_of_words = {}
        for flow_id in rem_ports_keywords:
            if flow_id not in rem_ports_bag_of_words:
                rem_ports_bag_of_words[flow_id] = {}
            for word in rem_ports_wordset:
                if word not in rem_ports_keywords[flow_id]:
                    rem_ports_bag_of_words[flow_id][word] = 0
                else:
                    rem_ports_bag_of_words[flow_id][word] = 1
                    
        domain_bag_of_words = {}
        for flow_id in domain_keywords:
            if flow_id not in domain_bag_of_words:
                domain_bag_of_words[flow_id] = {}
            for word in domain_wordset:
                if word not in domain_keywords[flow_id]:
                    domain_bag_of_words[flow_id][word] = 0
                else:
                    domain_bag_of_words[flow_id][word] = 1
                    
        rem_ports_bag_of_words_df = pd.DataFrame.from_dict(rem_ports_bag_of_words, orient='index')
        domain_bag_of_words_df = pd.DataFrame.from_dict(domain_bag_of_words, orient='index')
        
        return (device_flow_id_map, rem_ports_bag_of_words_df, domain_bag_of_words_df)

def main():
    #fill_domain("sample2.csv")
    #print(generate_bag_of_words("sample2.csv"))
    pass

if __name__ == "__main__":
    main()
