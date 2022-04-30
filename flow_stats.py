from xml import dom
import pyshark
import pandas as pd

#TODO: for live capture, use pyshark.LiveCapture


def add_to_flow_dict(flow_dict,flow_id,pkt):
    if flow_id not in flow_dict:
        flow_dict[flow_id] = []
    if flow_id in flow_dict:
        flow_dict[flow_id].append(pkt)


def flow_statistics(filename):
    udp_flow_dict = {}
    tcp_flow_dict = {}
    target_macs = []
    with open("maclist.txt") as f:
        target_macs = [line.rstrip().lower() for line in f]
    #Take count of packets
    total = 0
    udp = 0
    tcp = 0

    cap = pyshark.FileCapture(filename)


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
        except:
            pass
    
    flow_stats = {}
    mac_sleep_time = {}

    for flow_id in tcp_flow_dict:
        tcp_flow_dict[flow_id] = sorted(tcp_flow_dict[flow_id], key = lambda x:x.sniff_timestamp)
    for flow_id in udp_flow_dict:
        udp_flow_dict[flow_id] = sorted(udp_flow_dict[flow_id], key = lambda x:x.sniff_timestamp)

    for packet_dict in [tcp_flow_dict,udp_flow_dict]:
        for idx in range(len(packet_dict)):
            if packet_dict == tcp_flow_dict:
                prefix = "t"
            else:
                prefix = "u"
            
            if idx not in packet_dict:
                continue

            id = prefix + str(idx)
            # print("Parsing new flow...")
            total_len = 0
            total_pkt = 0
            start_duration = 0
            end_duration = 0

            dns_times = []
            domain = None
            for packet in packet_dict[idx]:
                if total_pkt == 0:
                    start_duration = float(packet.sniff_timestamp)
                elif total_pkt == len(packet_dict[idx])-1:
                    end_duration = float(packet.sniff_timestamp)
                if packet.highest_layer == "DNS" or packet.highest_layer == "MDNS":
                    if packet.highest_layer == "DNS":
                        domain = packet.dns.qry_name
                    else:
                        if "dns_qry_name" in dir(packet.mdns):
                            domain = packet.mdns.dns_qry_name
                        else:
                            domain = None
                    if packet.sniff_timestamp.split('.')[0] not in dns_times:
                        dns_times.append(packet.sniff_timestamp.split('.')[0])

                total_len += int(packet.length)
                total_pkt += 1
            if (packet.eth.dst.lower() not in target_macs and packet.eth.src.lower() not in target_macs):
                # print(packet.eth.dst)
                continue 
            flow_stats[id] = {}
            
            flow_stats[id]["source_port"] = packet[packet.transport_layer].srcport
            flow_stats[id]["dest_port"] = packet[packet.transport_layer].dstport
            try:
                flow_stats[id]["source_ip"] = packet.ip.src
                flow_stats[id]["dest_ip"] = packet.ip.dst
            except AttributeError:
                flow_stats[id]["source_ip"] = packet.ipv6.src
                flow_stats[id]["dest_ip"] = packet.ipv6.dst
            flow_stats[id]["source_mac"] = packet.eth.src
            flow_stats[id]["dest_mac"] = packet.eth.dst
            
            if packet.eth.src not in mac_sleep_time:
                mac_sleep_time[packet.eth.src] = [float(packet.sniff_timestamp)]
            else:
                mac_sleep_time[packet.eth.src].append(float(packet.sniff_timestamp))

            if packet.eth.dst not in mac_sleep_time:
                mac_sleep_time[packet.eth.dst] = [float(packet.sniff_timestamp)]
            else:
                mac_sleep_time[packet.eth.dst].append(float(packet.sniff_timestamp))
                
            if len(packet_dict[idx]) == 1:
                flow_stats[id]["flow_duration"] = 0
                flow_stats[id]["flow_rate"] = total_pkt
            else:
                flow_stats[id]["flow_duration"] = end_duration - start_duration
                flow_stats[id]["flow_rate"] = total_pkt/flow_stats[id]["flow_duration"]
            
            flow_stats[id]["flow_volume"] = total_len
            flow_stats[id]["domain"] = domain
            print ("Added flow " , id)
    cap.close()
    
    mac_diff = {}

    #sort timestamp list of each mac
    for mac in mac_sleep_time:
        mac_sleep_time[mac] = sorted(mac_sleep_time[mac])
        
    for mac in mac_sleep_time:
        mac_diff[mac] = [sleep_time - mac_sleep_time[mac][i - 1] for i, sleep_time in enumerate(mac_sleep_time[mac])][1:]

    mac_avg_sleep_time = {}
    for mac in mac_diff:
        if len(mac_diff[mac]) > 0:
            mac_avg_sleep_time[mac] = sum(mac_diff[mac])/len(mac_diff[mac])
        else:
            mac_avg_sleep_time[mac] = 0

#     mac_max_sleep_time = {}
#     for mac in mac_diff:
#         if len(mac_diff[mac]) > 0:
#             mac_max_sleep_time[mac] = max(mac_diff[mac])
#         else:
#             mac_max_sleep_time[mac] = 0

#     print (mac_sleep_time)
#     print (mac_diff)
    print (mac_avg_sleep_time)
#     print (mac_max_sleep_time)

    return flow_stats

#flow_stats = flow_statistics("sample2.pcap")


