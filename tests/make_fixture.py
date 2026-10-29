"""Generate synthetic packets for integration checks; never use for reported results."""
from pathlib import Path
import sys
from scapy.all import Ether, IP, UDP, Raw, wrpcap


def make(root):
    root = Path(root)
    families = [('code_data_engineering', 15), ('general_reasoning_qa', 21),
                ('web_search_information_retrieval', 28), ('conversational_service_agent', 20),
                ('attack_execution', 40)]
    for si, split in enumerate(['train', 'val', 'test']):
        for ci, (family, task) in enumerate(families):
            for sample in range(2):
                episode = f'A{task:02d}{si * 10 + sample:02d}'
                path = root / split / 'fixture_engine' / family / episode / 'network.pcap'
                path.parent.mkdir(parents=True, exist_ok=True)
                packets = []
                for flow in range(4):
                    for k in range(12):
                        src, dst = '10.0.0.1', f'10.0.{ci + 1}.{flow + 2}'
                        sport, dport = 20000 + flow, 443
                        if k % 2:
                            src, dst, sport, dport = dst, src, dport, sport
                        packet = Ether(src='02:00:00:00:00:01', dst='02:00:00:00:00:02') / IP(src=src, dst=dst) / UDP(sport=sport, dport=dport) / Raw(bytes([65 + ci]) * (50 + ci * 40 + sample))
                        packet.time = 1700000000 + flow * 2 + k * (0.01 + ci * 0.001)
                        packets.append(packet)
                wrpcap(str(path), packets)


if __name__ == '__main__':
    make(sys.argv[1])
