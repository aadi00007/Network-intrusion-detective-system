"""Scapy-based live packet capture that emits NSL-KDD feature rows.

This script approximates NSL-KDD connection features from live network
traffic, then optionally runs the bundled sklearn pipeline to classify each
connection. Many higher-level NSL-KDD statistics (e.g. `serror_rate`) require
extended session state across time windows. We provide simple heuristics so the
output aligns with the expected 41-feature schema, while keeping the code
understandable and self-contained.

Usage example (requires root privileges to sniff packets):

    sudo ./venv/bin/python live_capture.py sniff --iface en0 --timeout 10 \
        --model_path models/nsl_kdd_model.joblib \
        --label_map_path models/label_map.joblib \
        --output_csv tmp_predictions.csv

The script will:
1. Capture packets on the specified interface for the requested duration or
   packet count.
2. Aggregate packets into bi-directional flows.
3. Approximate the 41 NSL-KDD numeric/categorical features for each flow.
4. Optionally run the trained IDS model to classify the generated rows.

Limitations:
- Only IPv4 TCP/UDP traffic is interpreted. Other protocols are skipped.
- Advanced statistical features (e.g. *_rate columns) use capture-local
  heuristics rather than full dataset semantics.
- Service detection relies on a small port→name lookup with a fallback.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import load


NSL_KDD_COLUMNS: List[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# Common TCP/UDP service names used in NSL-KDD.
SERVICE_PORT_MAP: Mapping[int, str] = {
    20: "ftp_data",
    21: "ftp",
    22: "ssh",
    23: "telnet",
    25: "smtp",
    42: "name",
    43: "whois",
    53: "domain",
    67: "dhcp",
    68: "dhcp",
    69: "tftp_u",
    70: "gopher",
    79: "finger",
    80: "http",
    109: "pop_2",
    110: "pop_3",
    111: "rpc",
    113: "auth",
    119: "nntp",
    123: "ntp_u",
    135: "msrpc",
    137: "netbios_ns",
    138: "netbios_dgm",
    139: "netbios_ssn",
    143: "imap4",
    161: "snmp_u",
    162: "snmptrap_u",
    179: "bgp",
    194: "irc",
    389: "ldap",
    443: "https",
    445: "microsoft_ds",
    465: "ssmtp",
    500: "isakmp",
    514: "shell",
    520: "route",
    548: "afp",
    554: "rtsp",
    623: "asf_rmcp",
    993: "imap4_ssl",
    995: "pop3_ssl",
    1080: "socks",
    1433: "ms_sql_s",
    1521: "oracle",
    1723: "pptp",
    1900: "upnp",
    2049: "nfs",
    3306: "mysql",
    3389: "ms_term_serv",
    5060: "sip",
    5432: "postgres",
    5900: "vnc",
    6379: "redis",
    8080: "http_proxy",
    8443: "https_alt",
}


def resolve_service(port: int) -> str:
    """Return a best-effort NSL-KDD service name for the destination port."""
    return SERVICE_PORT_MAP.get(port, "other")


def derive_flag(forward_flags: Counter[str], reverse_flags: Counter[str]) -> str:
    """Approximate NSL-KDD connection state flags from TCP conversation."""
    if not forward_flags and not reverse_flags:
        return "OTH"

    total_flags = Counter(forward_flags)
    total_flags.update(reverse_flags)
    if total_flags.get("R", 0) > 0:
        if not reverse_flags:
            return "REJ"
        return "RSTO"
    if total_flags.get("S", 0) > 0 and total_flags.get("F", 0) > 0:
        return "SH"
    if total_flags.get("S", 0) > 0 and total_flags.get("A", 0) == 0:
        return "S0"
    return "SF"


@dataclass
class FlowRecord:
    """Represents a bi-directional connection aggregate."""

    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    first_seen: float
    last_seen: float
    src_bytes: int = 0
    dst_bytes: int = 0
    src_packets: int = 0
    dst_packets: int = 0
    land: int = 0
    wrong_fragment: int = 0
    urgent: int = 0
    forward_flags: Counter[str] = field(default_factory=Counter)
    reverse_flags: Counter[str] = field(default_factory=Counter)

    def update(
        self,
        timestamp: float,
        packet_len: int,
        direction: str,
        tcp_flags: Optional[str],
        wrong_fragment: int,
        urgent: int,
    ) -> None:
        self.last_seen = max(self.last_seen, timestamp)
        if direction == "forward":
            self.src_bytes += packet_len
            self.src_packets += 1
            if tcp_flags:
                self.forward_flags.update(tcp_flags)
        else:
            self.dst_bytes += packet_len
            self.dst_packets += 1
            if tcp_flags:
                self.reverse_flags.update(tcp_flags)
        self.wrong_fragment += wrong_fragment
        self.urgent += urgent

    def duration(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)


class FlowAccumulator:
    """Aggregates scapy packets into NSL-KDD feature rows."""

    def __init__(self) -> None:
        self._flows: Dict[Tuple[str, str, int, int, str], FlowRecord] = {}
        self._aliases: Dict[Tuple[str, str, int, int, str], Tuple[str, str, int, int, str]] = {}

    def process_packet(self, packet) -> None:
        """Update aggregates for a single Scapy packet."""
        # Lazy import to keep tests fast when Scapy is unavailable.
        from scapy.layers.inet import IP, TCP, UDP

        if not packet.haslayer(IP):
            return
        ip_layer = packet[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        protocol = "tcp" if packet.haslayer(TCP) else "udp" if packet.haslayer(UDP) else None
        if protocol is None:
            return

        src_port = int(packet[TCP].sport) if protocol == "tcp" else int(packet[UDP].sport)
        dst_port = int(packet[TCP].dport) if protocol == "tcp" else int(packet[UDP].dport)

        key = (src_ip, dst_ip, src_port, dst_port, protocol)
        reverse_key = (dst_ip, src_ip, dst_port, src_port, protocol)
        canonical = self._aliases.get(key, key)

        record = self._flows.get(canonical)
        direction = "forward"
        if record is None:
            record = FlowRecord(
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=protocol,
                first_seen=float(packet.time),
                last_seen=float(packet.time),
                land=int(src_ip == dst_ip),
            )
            self._flows[key] = record
            self._aliases[key] = key
            self._aliases[reverse_key] = key
            canonical = key
        else:
            if canonical != key:
                direction = "reverse"

        packet_len = int(len(packet))
        tcp_flags = None
        urgent = 0
        wrong_fragment = 0

        if protocol == "tcp" and packet.haslayer(TCP):
            tcp_layer = packet[TCP]
            tcp_flags = tcp_layer.sprintf("%TCP.flags%")
            urgent = int(bool(tcp_layer.flags & 0x20))

        if packet.haslayer(IP):
            wrong_fragment = int(bool(getattr(ip_layer, "frag", 0)))

        record.update(
            timestamp=float(packet.time),
            packet_len=packet_len,
            direction=direction,
            tcp_flags=tcp_flags,
            wrong_fragment=wrong_fragment,
            urgent=urgent,
        )

    def iter_records(self) -> Iterable[FlowRecord]:
        return self._flows.values()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert accumulated flows into a NSL-KDD shaped DataFrame."""
        records = list(self.iter_records())
        if not records:
            return pd.DataFrame(columns=NSL_KDD_COLUMNS)

        per_src_dst: Counter[Tuple[str, str]] = Counter()
        per_src_srv: Counter[Tuple[str, int]] = Counter()
        per_dst: Counter[str] = Counter()
        per_dst_srv: Counter[Tuple[str, int]] = Counter()
        per_dst_src_port: Counter[Tuple[str, int]] = Counter()

        for rec in records:
            per_src_dst[(rec.src_ip, rec.dst_ip)] += 1
            per_src_srv[(rec.src_ip, rec.dst_port)] += 1
            per_dst[rec.dst_ip] += 1
            per_dst_srv[(rec.dst_ip, rec.dst_port)] += 1
            per_dst_src_port[(rec.dst_ip, rec.src_port)] += 1

        rows: List[List[float | int | str]] = []
        for rec in records:
            count = per_src_dst[(rec.src_ip, rec.dst_ip)]
            srv_count = per_src_srv[(rec.src_ip, rec.dst_port)]
            dst_host_count = per_dst[rec.dst_ip]
            dst_host_srv_count = per_dst_srv[(rec.dst_ip, rec.dst_port)]
            dst_host_same_src_port_count = per_dst_src_port[(rec.dst_ip, rec.src_port)]

            same_srv_rate = srv_count / count if count else 0.0
            diff_srv_rate = max(0.0, 1.0 - same_srv_rate)
            srv_diff_host_rate = max(
                0.0,
                1.0 - (dst_host_srv_count / dst_host_count if dst_host_count else 0.0),
            )
            dst_host_same_srv_rate = dst_host_srv_count / dst_host_count if dst_host_count else 0.0
            dst_host_diff_srv_rate = max(0.0, 1.0 - dst_host_same_srv_rate)
            dst_host_same_src_port_rate = (
                dst_host_same_src_port_count / dst_host_count if dst_host_count else 0.0
            )

            flag = derive_flag(rec.forward_flags, rec.reverse_flags)
            service = resolve_service(rec.dst_port)

            row = [
                rec.duration(),  # duration
                rec.protocol,  # protocol_type
                service,  # service
                flag,  # flag
                rec.src_bytes,
                rec.dst_bytes,
                rec.land,
                rec.wrong_fragment,
                rec.urgent,
                0,  # hot
                0,  # num_failed_logins
                0,  # logged_in
                0,  # num_compromised
                0,  # root_shell
                0,  # su_attempted
                0,  # num_root
                0,  # num_file_creations
                0,  # num_shells
                0,  # num_access_files
                0,  # num_outbound_cmds
                0,  # is_host_login
                0,  # is_guest_login
                count,
                srv_count,
                0.0,  # serror_rate
                0.0,  # srv_serror_rate
                0.0,  # rerror_rate
                0.0,  # srv_rerror_rate
                same_srv_rate,
                diff_srv_rate,
                srv_diff_host_rate,
                dst_host_count,
                dst_host_srv_count,
                dst_host_same_srv_rate,
                dst_host_diff_srv_rate,
                dst_host_same_src_port_rate,
                dst_host_diff_srv_rate,  # reuse as approximation
                0.0,  # dst_host_serror_rate
                0.0,  # dst_host_srv_serror_rate
                0.0,  # dst_host_rerror_rate
                0.0,  # dst_host_srv_rerror_rate
            ]
            rows.append(row)

        df = pd.DataFrame(rows, columns=NSL_KDD_COLUMNS)
        numeric_cols = [col for col in NSL_KDD_COLUMNS if df[col].dtype != "O"]
        df[numeric_cols] = df[numeric_cols].fillna(0.0)
        return df


def sniff_packets(
    iface: Optional[str],
    count: Optional[int],
    timeout: Optional[int],
) -> FlowAccumulator:
    """Capture live packets using Scapy and aggregate them."""
    try:
        from scapy.all import sniff
    except ImportError as exc:
        raise SystemExit(
            "Scapy is required for live capture. Install it with `pip install scapy`."
        ) from exc

    accumulator = FlowAccumulator()

    def _handle(packet):
        try:
            accumulator.process_packet(packet)
        except Exception as err:
            logging.getLogger(__name__).exception("Error handling packet: %s", err)

    sniff_kwargs = {
        "iface": iface,
        "prn": _handle,
        "filter": "tcp or udp",
        "store": False,
    }
    if count:
        sniff_kwargs["count"] = count
    if timeout:
        sniff_kwargs["timeout"] = timeout

    sniff(**sniff_kwargs)
    return accumulator


def classify_dataframe(
    df: pd.DataFrame,
    model_path: Optional[Path],
    label_map_path: Optional[Path],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if not model_path or not label_map_path:
        return pd.DataFrame()

    pipeline = load(model_path)
    label_info = load(label_map_path)
    classes = label_info["classes_"]

    features = df.iloc[:, :41]
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        probs = pipeline.predict_proba(features)
        preds = np.argmax(probs, axis=1)
        confidence = probs.max(axis=1)
    else:
        preds = pipeline.predict(features)
        confidence = np.ones(len(preds))

    labels = classes[preds]
    result = df.copy()
    result["predicted_label"] = labels
    result["confidence"] = confidence
    return result


def write_output(df: pd.DataFrame, output_csv: Optional[Path]) -> None:
    if not output_csv:
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logging.info("Wrote captured features to %s", output_csv)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture live packets and classify them.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sniff_parser = subparsers.add_parser("sniff", help="Capture packets from a network interface.")
    sniff_parser.add_argument("--iface", default=None, help="Network interface to sniff (default: scapy chooses).")
    sniff_parser.add_argument("--count", type=int, default=None, help="Stop after capturing N packets.")
    sniff_parser.add_argument("--timeout", type=int, default=None, help="Time in seconds to capture packets.")
    sniff_parser.add_argument("--model_path", type=Path, default=None, help="Path to trained joblib pipeline.")
    sniff_parser.add_argument("--label_map_path", type=Path, default=None, help="Path to label map joblib.")
    sniff_parser.add_argument("--output_csv", type=Path, default=None, help="Write captured feature rows to CSV.")
    sniff_parser.add_argument("--predict", action="store_true", help="Run the classifier if model artifacts are provided.")

    return parser.parse_args(argv)


def command_sniff(args: argparse.Namespace) -> int:
    logging.info(
        "Starting capture iface=%s count=%s timeout=%s",
        args.iface,
        args.count,
        args.timeout,
    )
    accumulator = sniff_packets(args.iface, args.count, args.timeout)
    df = accumulator.to_dataframe()
    if df.empty:
        logging.warning("No flows captured.")
    else:
        logging.info("Captured %d flow(s).", len(df))
    write_output(df, args.output_csv)

    if args.predict and args.model_path and args.label_map_path:
        logging.info("Running classifier using %s", args.model_path)
        predictions = classify_dataframe(df, args.model_path, args.label_map_path)
        if predictions.empty:
            logging.warning("Prediction output is empty.")
        else:
            for idx, row in predictions.iterrows():
                logging.info(
                    "Flow %d → label=%s confidence=%.3f service=%s protocol=%s",
                    idx,
                    row["predicted_label"],
                    row["confidence"],
                    row["service"],
                    row["protocol_type"],
                )
        if args.output_csv:
            predictions.to_csv(args.output_csv, index=False)
            logging.info("Wrote predictions to %s", args.output_csv)
    elif args.predict:
        logging.warning("Prediction requested but model or label map path is missing.")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args(argv)
    if args.command == "sniff":
        return command_sniff(args)
    logging.error("Unknown command: %s", args.command)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

