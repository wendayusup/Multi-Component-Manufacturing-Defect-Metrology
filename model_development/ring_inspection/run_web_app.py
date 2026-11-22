"""
QUICK LAUNCHER: STAGE 6 WEB INSPECTION DASHBOARD
------------------------------------------------
Launches the Flask web server for industrial bearing defect inspection.
"""

import sys
import socket
from step06_web_app.app import app
from step06_web_app.ssl_util import get_or_create_ssl_cert, get_local_ip

if __name__ == "__main__":
    use_https = "--https" in sys.argv
    local_ip = get_local_ip()

    ssl_context = None
    protocol = "http"

    if use_https:
        certs = get_or_create_ssl_cert()
        if certs:
            ssl_context = certs
            protocol = "https"
        else:
            print("[WARN] SSL certificate could not be created. Falling back to HTTP.")

    print("\n" + "=" * 68)
    print("      BEARING DEFECT INSPECTION WEB APPLICATION RUNNING")
    print("=" * 68)
    if use_https:
        print("   Mode: [HTTPS - SECURE FOR MOBILE CAMERAS]")
        print(f"   💻 Laptop Browser:    https://localhost:5000")
        print(f"   📱 Mobile (Same Wi-Fi): https://{local_ip}:5000")
        print("   (Di browser HP klik 'Advanced' -> 'Proceed' untuk izinkan kamera)")
    else:
        print("   Mode: [HTTP - STANDARD LOCAL]")
        print("   💻 Laptop Browser:    http://localhost:5000")
        print(f"   📱 Mobile (Same Wi-Fi): http://{local_ip}:5000")
        print("\n   💡 TIPS AKSES KAMERA HP:")
        print("      Jalankan dengan flag HTTPS agar browser HP langsung izinkan kamera:")
        print("      python run_web_app.py --https")
    print("=" * 68)
    print("   Press CTRL+C in terminal to stop server.\n")

    if ssl_context:
        app.run(host="0.0.0.0", port=5000, debug=False, ssl_context=ssl_context)
    else:
        app.run(host="0.0.0.0", port=5000, debug=False)
