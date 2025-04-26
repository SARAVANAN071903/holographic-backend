import torch
import torch.nn as nn
import numpy as np
import cv2
import qrcode
import os
from PIL import Image

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(100, 128*self.init_size**2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

def generate_pure_qr(otp_text, size=512):
    """Generate high-contrast QR code without visible text"""
    qr = qrcode.QRCode(
        version=4,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=12,
        border=6,
        mask_pattern=2
    )
    qr.add_data(otp_text)
    qr.make(fit=True)
    return np.array(qr.make_image(
        fill_color=(0, 0, 0),  # Pure black modules
        back_color=(255, 255, 255)  # Pure white background
    ).convert('RGB'))

def generate_holographic_otp(otp_text: str, output_dir: str):
    """Generate secure holographic OTP image"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"holographic_otp_{otp_text}.png")

        # 1. Load GAN model
        generator = Generator()
        model_path = os.path.join(os.path.dirname(__file__), "dcgan_generator_final.pth")
        generator.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        generator.eval()

        # 2. Generate hologram
        with torch.no_grad():
            noise = torch.randn(1, 100)
            hologram = generator(noise).squeeze().numpy()
            hologram = ((hologram + 1) * 127.5).astype(np.uint8)
            hologram = cv2.cvtColor(hologram, cv2.COLOR_GRAY2BGR)
            hologram = cv2.GaussianBlur(hologram, (3, 3), 0)  # Reduce noise

        # 3. Generate optimized QR
        qr_img = generate_pure_qr(otp_text)
        qr_img = cv2.resize(qr_img, (hologram.shape[1], hologram.shape[0]))

        # 4. Perfect blending (45% hologram, 55% QR)
        blended = cv2.addWeighted(hologram, 0.45, qr_img, 0.55, 0)
        
        # 5. Enhance contrast for better scanning
        blended = cv2.convertScaleAbs(blended, alpha=1.1, beta=10)

        cv2.imwrite(output_path, blended)
        return output_path

    except Exception as e:
        raise RuntimeError(f"Generation failed: {str(e)}")

if __name__ == "__main__":
    try:
        path = generate_holographic_otp("TEST123", "../output")
        print(f"Successfully generated: {path}")
    except Exception as e:
        print(f"Error: {e}")