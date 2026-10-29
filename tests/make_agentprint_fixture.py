"""Create the T01 hand-checkable demo; synthetic data is not research evidence."""
from pathlib import Path
import sys

from scapy.all import Ether, IP, TCP, wrpcap


def make(root):
    root = Path(root)
    for split in ('train', 'val'):
        target = root / split / 'fixture_engine' / 'artifact_processing' / 'A001' / 'network.pcap'
        target.parent.mkdir(parents=True, exist_ok=True)
        packets = []
        for offset, src, dst, flags in (
                (0, '10.0.0.2', '8.8.8.8', 'S'),
                (1, '10.0.0.2', '1.1.1.1', 'S'),
                (900, '8.8.8.8', '10.0.0.2', 'SA'),
                (1800, '10.0.0.2', '8.8.8.8', 'A')):
            packet = (Ether(src='02:00:00:00:00:01', dst='02:00:00:00:00:02') /
                      IP(src=src, dst=dst) / TCP(sport=12345, dport=443, flags=flags))
            packet.time = 1700000000 + offset
            packets.append(packet)
        wrpcap(str(target), packets)


if __name__ == '__main__':
    make(sys.argv[1])
