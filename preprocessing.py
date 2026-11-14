"""MODULE 1 — LÊ SƠN HẢI: TIẾP NHẬN VÀ TIỀN XỬ LÝ DỮ LIỆU ẢNH UAV.

Chức năng:
    - Tiếp nhận ảnh đa định dạng (PNG, JPG, JPEG, TIF, TIFF).
    - Chuẩn hóa kênh màu RGB và kích thước chuẩn (512x512).
    - Chuyển đổi thành PyTorch Tensor và chuẩn hóa ImageNet (mean/std).

Cách chạy kiểm thử độc lập:
    python preprocessing.py --test
"""

import io
import sys
import unittest
from typing import Tuple, Union

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

# Hỗ trợ console Windows hiển thị tiếng Việt
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image(input_source: Union[str, bytes, io.BytesIO, Image.Image]) -> Image.Image:
    """Nạp ảnh từ nhiều nguồn dữ liệu khác nhau (Đường dẫn, BytesIO, hoặc đối tượng PIL)."""
    if isinstance(input_source, Image.Image):
        return input_source
    if isinstance(input_source, (str, bytes, io.BytesIO)):
        try:
            img = Image.open(input_source if not isinstance(input_source, bytes) else io.BytesIO(input_source))
            img.load()  # Nạp toàn bộ dữ liệu vào RAM
            return img
        except Exception as e:
            raise ValueError(f"Không thể đọc file ảnh: {e}")
    raise TypeError(f"Nguồn ảnh không hợp lệ: {type(input_source)}")


def preprocess_image(
    pil_image: Image.Image,
    target_size: Tuple[int, int] = (512, 512)
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Tiền xử lý ảnh tải lên phục vụ nạp vào mạng nơ-ron U-Net.
    
    Args:
        pil_image: Đối tượng ảnh PIL gốc.
        target_size: Kích thước pixel chuẩn của mô hình (W, H), mặc định (512, 512).
        
    Returns:
        img_t: Tensor 4 chiều (1, 3, 512, 512) đã chuẩn hóa ImageNet.
        (orig_w, orig_h): Kích thước pixel ban đầu của ảnh để Module 3 và 4 quy đổi.
    """
    if not isinstance(pil_image, Image.Image):
        raise TypeError(f"Đầu vào phải là đối tượng PIL.Image, nhận được {type(pil_image)}")

    orig_w, orig_h = pil_image.size

    # Chuẩn hóa về hệ màu RGB (loại bỏ kênh Alpha nếu là PNG/TIFF)
    img_rgb = pil_image.convert("RGB")

    # Resize song tuyến (Bilinear) về kích thước chuẩn huấn luyện
    img_resized = img_rgb.resize(target_size, Image.BILINEAR)

    # Chuyển đổi sang PyTorch Tensor [0.0, 1.0]
    img_t = TF.to_tensor(img_resized)

    # Chuẩn hóa Normalize theo phân phối chuẩn ImageNet
    img_t = TF.normalize(img_t, mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0)

    return img_t, (orig_w, orig_h)


# ==============================================================================
# BỘ KIỂM THỬ ĐỘC LẬP (UNIT TESTS) — Chạy bằng: python preprocessing.py --test
# ==============================================================================

class TestPreprocessing(unittest.TestCase):
    """Kiểm thử unit test độc lập cho Module 1."""

    def test_preprocess_rgb_image(self):
        """Ảnh RGB thông thường kích thước 800x600 phải được chuyển thành Tensor (1, 3, 512, 512)."""
        img = Image.new("RGB", (800, 600), color=(120, 150, 180))
        tensor, orig_size = preprocess_image(img, target_size=(512, 512))
        self.assertEqual(orig_size, (800, 600))
        self.assertEqual(tensor.shape, (1, 3, 512, 512))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_preprocess_rgba_converts_to_3_channels(self):
        """Ảnh RGBA 4 kênh phải được tự động chuyển thành RGB 3 kênh."""
        img = Image.new("RGBA", (400, 400), color=(100, 200, 50, 128))
        tensor, _ = preprocess_image(img)
        self.assertEqual(tensor.shape[1], 3)

    def test_preprocess_grayscale_converts_to_3_channels(self):
        """Ảnh đen trắng L 1 kênh phải được chuyển thành RGB 3 kênh."""
        img = Image.new("L", (256, 256), color=128)
        tensor, _ = preprocess_image(img)
        self.assertEqual(tensor.shape[1], 3)

    def test_invalid_input_raises_error(self):
        """Đầu vào không phải PIL.Image phải báo lỗi TypeError."""
        with self.assertRaises(TypeError):
            preprocess_image("duong_dan_anh_gia.jpg")


def main():
    if "--test" in sys.argv:
        print("[Module 1] Đang chạy kiểm thử Unit Test cho Preprocessing...")
        sys.argv.remove("--test")
        unittest.main()
    else:
        print("=" * 60)
        print("MODULE 1 — LÊ SƠN HẢI: TIẾN TRÌNH TIỀN XỬ LÝ ẢNH UAV")
        print("=" * 60)
        dummy_img = Image.new("RGB", (1024, 768), color=(150, 180, 200))
        tensor, orig_dim = preprocess_image(dummy_img)
        print(f"Ảnh gốc: {orig_dim[0]}x{orig_dim[1]} px")
        print(f"Tensor chuẩn hóa: {tensor.shape}, Mean={tensor.mean():.4f}, Std={tensor.std():.4f}")
        print("Chạy 'python preprocessing.py --test' để kiểm thử độc lập.")


if __name__ == "__main__":
    main()
