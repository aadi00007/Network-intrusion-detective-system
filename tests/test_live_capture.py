import time

import pandas as pd
import pytest
from scapy.layers.inet import IP, TCP

from live_capture import FlowAccumulator


def build_packet(src, dst, sport, dport, flags, timestamp):
    pkt = IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags=flags)
    pkt.time = timestamp
    return pkt


def test_flow_accumulator_creates_single_flow():
    acc = FlowAccumulator()
    base = time.time()
    packets = [
        build_packet("10.0.0.1", "192.168.0.5", 12345, 80, "S", base),
        build_packet("192.168.0.5", "10.0.0.1", 80, 12345, "SA", base + 0.1),
        build_packet("10.0.0.1", "192.168.0.5", 12345, 80, "A", base + 0.2),
    ]

    for pkt in packets:
        acc.process_packet(pkt)

    df = acc.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1

    row = df.iloc[0]
    assert row["protocol_type"] == "tcp"
    assert row["service"] == "http"
    assert row["count"] == 1
    assert row["srv_count"] == 1
    assert row["src_bytes"] > 0
    assert row["dst_bytes"] > 0
    assert row["flag"] in {"SF", "S0"}
    assert pytest.approx(row["duration"], rel=1e-2) == 0.2

