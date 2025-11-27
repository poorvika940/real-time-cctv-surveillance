# Generate a self-signed development certificate (PEM) for HTTPS
# Usage:
#   .\.venv\Scripts\python.exe scripts\generate_dev_cert.py --hosts localhost,127.0.0.1,192.168.175.38 --out models\certs
import argparse
import os
import ipaddress
from datetime import datetime, timedelta
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def build_cert(hosts, days=3650):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, u"US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"Dev"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, u"Local"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Dev Cert"),
        x509.NameAttribute(NameOID.COMMON_NAME, u"Local Dev")
    ])
    alt_names = []
    for h in hosts:
        h = h.strip()
        if not h:
            continue
        try:
            alt_names.append(x509.IPAddress(ipaddress.ip_address(h)))
            continue
        except Exception:
            pass
        alt_names.append(x509.DNSName(h))
    san = x509.SubjectAlternativeName(alt_names)
    now = datetime.utcnow()
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=days))
        .add_extension(san, critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    return key, cert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hosts', default='localhost,127.0.0.1', help='Comma-separated hostnames/IPs for SAN')
    ap.add_argument('--out', default=os.path.join('models', 'certs'))
    args = ap.parse_args()
    hosts = [h.strip() for h in args.hosts.split(',') if h.strip()]
    os.makedirs(args.out, exist_ok=True)
    key, cert = build_cert(hosts)
    key_path = os.path.join(args.out, 'key.pem')
    cert_path = os.path.join(args.out, 'cert.pem')
    with open(key_path, 'wb') as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))
    with open(cert_path, 'wb') as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    print('Wrote cert:', cert_path)
    print('Wrote key :', key_path)


if __name__ == '__main__':
    main()
