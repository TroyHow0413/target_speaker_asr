"""Create a self-signed HTTPS certificate so the tablet browser allows microphone access.

    python scripts/make_cert.py                 # auto-detects this PC's LAN IP
    python scripts/make_cert.py 192.168.1.50    # or give IPs / hostnames explicitly
"""
import argparse
import datetime
import ipaddress
import socket
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

ROOT = Path(__file__).resolve().parent.parent


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hosts", nargs="*", help="extra IPs / hostnames to include")
    ap.add_argument("--out", default=str(ROOT / "certs"))
    args = ap.parse_args()

    hosts = ["localhost", "127.0.0.1", lan_ip(), socket.gethostname(), *args.hosts]
    sans, seen = [], set()
    for h in hosts:
        if h in seen:
            continue
        seen.add(h)
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            sans.append(x509.DNSName(h))

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "target-speaker-asr")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=825))
            .add_extension(x509.SubjectAlternativeName(sans), critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "server.key").write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
    (out / "server.crt").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    print(f"Wrote {out / 'server.crt'} and {out / 'server.key'}")
    print("Certificate covers:", ", ".join(sorted(seen)))
    print(f"Open on the tablet:  https://{lan_ip()}:8000   (accept the certificate warning once)")


if __name__ == "__main__":
    main()
