import pyshark
cap = pyshark.FileCapture('sample.pcap')
#TODO: for live capture, use pyshark.LiveCapture
sess_index = [] # to save stream indexes in an array
flow_dict = {}
for pkt in cap:
    try:
        # TODO: capturing only for udp stream - need to do for other streams like tcp
        sess_index.append(pkt.udp.stream)
        flow_id = int (pkt.udp.stream)
        if flow_id not in flow_dict:
            flow_dict[flow_id] = []
        if flow_id in flow_dict:
             flow_dict[flow_id].append(pkt)
    except:
        pass

# TODO: need to get for all flow statistics

flow_stats = {}
for idx in range(len(flow_dict)):
    total_len = 0
    total_pkt = 0
    for packet in flow_dict[idx]:
        total_len += int(packet.length)
        total_pkt += 1
    flow_stats[idx] = {}
    flow_stats[idx]["avg_len"] = total_len/total_pkt
    flow_stats[idx]["total_pkt"] = total_pkt
    flow_stats[idx]["total_len"] = total_len

print (flow_stats[3])

    #print (packet.length)
    #print (packet.transport_layer)
    #print (packet.ip.src)
    #print (packet.ip.dst)
    #print (packet[packet.transport_layer].srcport)
    #print (packet[packet.transport_layer].dstport)
    #print (packet.eth.src)
    #print (packet.eth.dst)
    #print ("###################")

#print (sess_index)
#if len(sess_index) == 0:
#    max_index = 0
#    print ("No TCP Found")
#else:
#    max_index = int(max(sess_index)) + 1 # max function is used to get the highiest number
#print (max_index)
