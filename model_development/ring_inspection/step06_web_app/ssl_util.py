"""
SSL UTILITY FOR LOCAL INDUSTRIAL WEB INSPECTION
------------------------------------------------
Automatically generates and loads self-signed TLS certificates for local HTTPS.
Enables camera access (WebRTC getUserMedia) across mobile phones and external devices.
"""

from pathlib import Path
import datetime
import socket

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def get_or_create_ssl_cert(cert_dir=None):
    """
    Ensures a valid self-signed certificate and private key exist for localhost & local IP.
    Returns (cert_path, key_path) or None if cryptography is unavailable.
    """
    if cert_dir is None:
        cert_dir = Path(__file__).resolve().parent / "certs"
    else:
        cert_dir = Path(cert_dir)

    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_path = cert_dir / "cert.pem"
    key_path = cert_dir / "key.pem"

    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)

    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import ipaddress

        # Generate RSA private key
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        local_ip = get_local_ip()

        # Subject & Issuer
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Politeknik Manufaktur Bandung"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])

        # SANs: localhost, 127.0.0.1, and local IP
        san_list = [
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        ]
        try:
            san_list.append(x509.IPAddress(ipaddress.IPv4Address(local_ip)))
        except Exception:
            pass

        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
            .sign(key, hashes.SHA256())
        )

        # Write key
        with open(key_path, "wb") as f:
            f.write(
                key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

        # Write cert
        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        print(f"[INFO] SSL Certificate auto-generated at: {cert_dir}")
        return str(cert_path), str(key_path)

    except Exception as e:
        print(f"[WARN] Could not auto-generate SSL certificate: {e}")
        return None
