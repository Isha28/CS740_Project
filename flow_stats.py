import pyshark
cap = pyshark.FileCapture('sample.pcap')
#TODO: for live capture, use pyshark.LiveCapture

udp_flow_dict = {}
tcp_flow_dict = {}

#Take count of packets
total = 0
udp = 0
tcp = 0

def add_to_flow_dict(flow_dict,flow_id,pkt):
    if flow_id not in flow_dict:
        flow_dict[flow_id] = []
    if flow_id in flow_dict:
        flow_dict[flow_id].append(pkt)

def flow_statistics(flow_dict):
    flow_stats = {}

    for idx in range(len(flow_dict)):
        total_len = 0
        total_pkt = 0
        start_duration = 0
        end_duration = 0

        for packet in flow_dict[idx]:
            if total_pkt == 0:
                start_duration = float(packet.sniff_timestamp)
            elif total_pkt == len(flow_dict[idx])-1:
                end_duration = float(packet.sniff_timestamp)

            total_len += int(packet.length)
            total_pkt += 1

        flow_stats[idx] = {}
        flow_stats[idx]["flow_duration"] = end_duration - start_duration
        flow_stats[idx]["flow_rate"] = total_pkt/flow_stats[idx]["flow_duration"]
        flow_stats[idx]["flow_volume"] = total_len
        #print (flow_stats[idx])

        #flow_stats[idx]["avg_pkt_len"] = total_len / total_pkt
        #flow_stats[idx]["total_pkt"] = total_pkt
        #flow_stats[idx]["total_len"] = total_len


    return (len(flow_stats))

for pkt in cap:
    total += 1
    try:
        if "UDP" in pkt and pkt.udp.stream != 0:
            udp += 1
            flow_id = int (pkt.udp.stream)
            add_to_flow_dict(udp_flow_dict, flow_id, pkt)

        elif "TCP" in pkt and pkt.tcp.stream != 0:
            tcp += 1
            flow_id = int (pkt.tcp.stream)
            add_to_flow_dict(tcp_flow_dict, flow_id, pkt)

        # Other streams not needed - Follow stream disabled for others in Wireshark

    except:
        pass

print ("total packets ", total)
print ("udp packets ", udp)
print ("tcp packets ", tcp)

print ("UDP flow statistics ", flow_statistics(udp_flow_dict))
print ("TCP flow statistics ", flow_statistics(tcp_flow_dict))

#print (packet.length)
#print (packet.transport_layer)
#print (packet.ip.src)
#print (packet.ip.dst)
#print (packet[packet.transport_layer].srcport)
#print (packet[packet.transport_layer].dstport)
#print (packet.eth.src)
#print (packet.eth.dst)


