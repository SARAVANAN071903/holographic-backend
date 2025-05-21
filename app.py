import os
import secrets
import numpy as np
import cv2
import qrcode
from flask_cors import CORS
from flask import Flask, jsonify, send_from_directory
from PIL import Image
from dotenv import load_dotenv

# ✅ Load environment variables from a .env file (for local testing)
load_dotenv()

app = Flask(__name__)

# ✅ Secure secret key usage
app.secret_key = os.getenv("SECRET_KEY", "fallback_dev_key")

# ✅ Enable CORS
CORS(app)

# ✅ Output directory for QR images
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_holographic_qr(otp_code, output_path):
    """Generate holographic-styled QR code without visible OTP"""
    try:
        qr = qrcode.QRCode(
            version=4,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=8,
            border=4
        )
        qr.add_data(otp_code)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white").convert('RGB')

        # Simple rainbow holographic pattern
        hologram = np.zeros((qr_img.size[1], qr_img.size[0], 3), dtype=np.uint8)
        for i in range(hologram.shape[0]):
            hologram[i, :, 0] = np.linspace(0, 255, hologram.shape[1])  # Red
            hologram[i, :, 1] = np.linspace(255, 0, hologram.shape[1])  # Green
            hologram[i, :, 2] = 128                                      # Blue

        # Blend QR code with hologram
        qr_np = np.array(qr_img)
        blended = cv2.addWeighted(hologram, 0.7, qr_np, 0.3, 0)

        # Save result
        cv2.imwrite(output_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        return True
    except Exception as e:
        print(f"Holographic QR generation failed: {e}")
        return False

@app.route('/generate_otp')
def generate_otp():
    try:
        otp_code = secrets.token_hex(3)  # 6-digit hexadecimal OTP
        filename = f"holographic_otp_{otp_code}.png"
        image_path = os.path.join(OUTPUT_DIR, filename)

        if not generate_holographic_qr(otp_code, image_path):
            raise RuntimeError("Failed to generate holographic OTP")

        return jsonify({
            "status": "success",
            "otp": otp_code,
            "image_url": f"/otp_image/{filename}",
            "expires_in": 300  # For client-side handling
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/otp_image/<filename>')
def serve_otp_image(filename):
    try:
        return send_from_directory(OUTPUT_DIR, filename, mimetype='image/png')
    except FileNotFoundError:
        return jsonify({"error": "OTP expired"}), 404

if __name__ == '__main__':
    app.run(debug=True)
