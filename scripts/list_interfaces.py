"""
List network interfaces visible to Scapy.

Run this on Windows after installing Npcap to discover the correct
interface name to pass into the live-capture API.

Usage:
    python scripts/list_interfaces.py
"""

from scapy.arch import get_if_list, get_if_addr


def main() -> None:
    print(f"{'Interface name':<60} {'IP address':<20}")
    print("-" * 80)
    for name in get_if_list():
        try:
            addr = get_if_addr(name)
        except Exception:
            addr = "?"
        print(f"{name:<60} {addr:<20}")


if __name__ == "__main__":
    main()
