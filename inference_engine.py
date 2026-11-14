"""MODULE 2 — NGUYỄN TRỌNG HÙNG (TRƯỞNG NHÓM): LÕI SUY LUẬN VÀ PHÂN ĐOẠN HỌC SÂU.

Hỗ trợ nạp mô hình học sâu phân đoạn ngữ nghĩa mái nhà từ ảnh viễn thám UAV:
- Kiến trúc U-Net (BuildingSegmentationNet)
- Kiến trúc DeepLabV3+ ResNet-50 (được huấn luyện tối ưu trên tập dữ liệu WHU)
- Quản lý bộ nhớ đệm (Cache) và tự động nhận diện thiết bị GPU CUDA / CPU.

Cách chạy kiểm thử độc lập:
    python inference_engine.py --test
"""

import os
import sys
import time
import unittest
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

# Đảm bảo in tiếng Việt mượt mà trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import streamlit as st
    cache_decorator = st.cache_resource
except ImportError:
    def cache_decorator(func):
        return func


# ==============================================================================
# 1. KIẾN TRÚC MẠNG NƠ-RON TÍCH CHẬP U-NET (BuildingSegmentationNet)
# ==============================================================================

class DoubleConv(nn.Module):
    """Khối tích chập kép: (Conv2D -> BatchNorm -> ReLU) x 2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Khối giảm mẫu Encoder: MaxPool2D(2x2) -> DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Khối tăng mẫu Decoder: ConvTranspose2D(2x2) -> Nối Skip Connection -> DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_x != 0 or diff_y != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class BuildingSegmentationNet(nn.Module):
    """Mạng U-Net gọn nhẹ gồm 4 khối Encoder và 4 khối Decoder kết hợp Skip Connections."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2, bilinear: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ==============================================================================
# 2. KIẾN TRÚC MẠNG DEEPLABV3+ RESNET-50 (MÔ HÌNH NÂNG CAO ĐÃ HUẤN LUYỆN WHU)
# ==============================================================================

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.global_pool(x)
        x = self.conv(x)
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256, atrous_rates=(6, 12, 18)):
        super().__init__()
        rate1, rate2, rate3 = atrous_rates
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ),
            ASPPConv(in_channels, out_channels, rate1),
            ASPPConv(in_channels, out_channels, rate2),
            ASPPConv(in_channels, out_channels, rate3),
            ASPPPooling(in_channels, out_channels),
        ])
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        x = torch.cat(features, dim=1)
        return self.project(x)


class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=None, replace_stride_with_dilation=[False, False, True])
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x = self.stem(x)
        low_level_features = self.layer1(x)
        x = self.layer2(low_level_features)
        x = self.layer3(x)
        high_level_features = self.layer4(x)
        return low_level_features, high_level_features


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = ResNet50Backbone()
        self.aspp = ASPP(in_channels=2048, out_channels=256, atrous_rates=(6, 12, 18))
        self.low_level_projection = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            ConvBNReLU(in_channels=256 + 48, out_channels=256, kernel_size=3, padding=1),
            nn.Dropout(p=0.1),
            ConvBNReLU(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        low_level, high_level = self.backbone(x)
        high_level = self.aspp(high_level)
        low_level = self.low_level_projection(low_level)
        high_level = F.interpolate(high_level, size=low_level.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([high_level, low_level], dim=1)
        logits = self.decoder(x)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


# ==============================================================================
# 3. HÀM QUẢN LÝ BỘ NHỚ ĐỆM VÀ NẠP TRỌNG SỐ MÔ HÌNH
# ==============================================================================

@cache_decorator
def load_model(weights_path: str = "best_model_weights.pth") -> Tuple[nn.Module, torch.device]:
    """Khởi tạo mô hình và nạp trọng số đã huấn luyện với cơ chế Caching."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(weights_path):
        try:
            try:
                checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
            except Exception:
                checkpoint = torch.load(weights_path, map_location=device)

            state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Tự động nhận diện cấu trúc mô hình qua keys trong state_dict
            first_key = list(state_dict.keys())[0] if isinstance(state_dict, dict) else ""
            if "backbone" in first_key:
                model = DeepLabV3Plus(num_classes=2)
                model_name = "DeepLabV3+ (ResNet-50 WHU)"
            else:
                model = BuildingSegmentationNet(in_channels=3, num_classes=2)
                model_name = "U-Net (BuildingSegmentationNet)"

            model.load_state_dict(state_dict)
            print(f"[Module 2] Đã nạp thành công mô hình [{model_name}] từ '{weights_path}' trên [{device}].")
        except Exception as e:
            print(f"[Module 2 CẢNH BÁO] Không thể nạp weights '{weights_path}': {e}. Sử dụng mô hình U-Net mặc định.")
            model = BuildingSegmentationNet(in_channels=3, num_classes=2)
    else:
        print(f"[Module 2 THÔNG BÁO] Không tìm thấy '{weights_path}'. Sử dụng mô hình U-Net mặc định.")
        model = BuildingSegmentationNet(in_channels=3, num_classes=2)

    model.to(device)
    model.eval()
    return model, device


# ==============================================================================
# 4. HÀM SUY LUẬN VÀ PHÂN NGƯỠNG ĐỘNG (Thresholding)
# ==============================================================================

def run_inference(
    model: nn.Module,
    device: torch.device,
    img_tensor: torch.Tensor,
    threshold: float = 0.75
) -> np.ndarray:
    mask, _ = run_inference_with_details(model, device, img_tensor, threshold=threshold)
    return mask


def run_inference_with_details(
    model: nn.Module,
    device: torch.device,
    img_tensor: torch.Tensor,
    threshold: float = 0.75
) -> Tuple[np.ndarray, dict]:
    if not isinstance(img_tensor, torch.Tensor):
        raise TypeError(f"img_tensor phải là torch.Tensor, nhận được {type(img_tensor)}.")

    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)
    elif img_tensor.ndim != 4:
        raise ValueError(f"img_tensor phải có 3 hoặc 4 chiều (B, C, H, W), nhận được shape {img_tensor.shape}.")

    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold phải nằm trong khoảng [0.0, 1.0], nhận được {threshold}.")

    start_time = time.perf_counter()

    with torch.no_grad():
        logits = model(img_tensor.to(device))
        probs = torch.softmax(logits, dim=1)
        prob_building = probs[0, 1, :, :].cpu().numpy()

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    binary_mask = (prob_building >= threshold).astype(np.uint8)

    meta = {
        "prob_building": prob_building,
        "latency_ms": latency_ms,
        "fps": 1000.0 / latency_ms if latency_ms > 0 else 0.0,
        "threshold": threshold,
        "device": str(device)
    }

    return binary_mask, meta


# ==============================================================================
# 5. BỘ KIỂM THỬ ĐỘC LẬP (UNIT TESTS) — Chạy bằng: python inference_engine.py --test
# ==============================================================================

class TestInferenceEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.device = load_model(weights_path="best_model_weights.pth")

    def test_model_architecture_and_output_shape(self):
        dummy_input = torch.randn(1, 3, 512, 512, device=self.device)
        with torch.no_grad():
            output = self.model(dummy_input)
        self.assertEqual(output.shape, (1, 2, 512, 512))

    def test_run_inference_returns_valid_numpy_mask(self):
        dummy_input = torch.randn(1, 3, 512, 512)
        mask = run_inference(self.model, self.device, dummy_input, threshold=0.75)
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.ndim, 2)
        self.assertEqual(mask.shape, (512, 512))
        self.assertEqual(mask.dtype, np.uint8)
        unique_vals = np.unique(mask)
        self.assertTrue(all(val in (0, 1) for val in unique_vals))

    def test_threshold_sensitivity(self):
        dummy_input = torch.randn(1, 3, 256, 256)
        mask_low = run_inference(self.model, self.device, dummy_input, threshold=0.3)
        mask_high = run_inference(self.model, self.device, dummy_input, threshold=0.8)
        self.assertGreaterEqual(np.sum(mask_low), np.sum(mask_high))

    def test_invalid_input_tensor_raises_error(self):
        with self.assertRaises(ValueError):
            run_inference(self.model, self.device, torch.randn(512, 512))

    def test_invalid_threshold_raises_error(self):
        dummy_input = torch.randn(1, 3, 128, 128)
        with self.assertRaises(ValueError):
            run_inference(self.model, self.device, dummy_input, threshold=1.5)
        with self.assertRaises(ValueError):
            run_inference(self.model, self.device, dummy_input, threshold=-0.2)

    def test_latency_measurement(self):
        dummy_input = torch.randn(1, 3, 256, 256)
        mask, meta = run_inference_with_details(self.model, self.device, dummy_input)
        self.assertIn("latency_ms", meta)
        self.assertGreater(meta["latency_ms"], 0.0)
        self.assertIn("prob_building", meta)


def main():
    if "--test" in sys.argv:
        print("[Module 2] Đang chạy kiểm thử Unit Test độc lập...")
        sys.argv.remove("--test")
        unittest.main()
    else:
        print("=" * 60)
        print("MODULE 2 — NGUYỄN TRỌNG HÙNG: LÕI SUY LUẬN AI")
        print("=" * 60)
        model, device = load_model("best_model_weights.pth")
        dummy_input = torch.randn(1, 3, 512, 512)
        mask, meta = run_inference_with_details(model, device, dummy_input, threshold=0.75)
        print(f"Thành công! Thời gian phản hồi: {meta['latency_ms']:.2f} ms (~{meta['fps']:.1f} FPS)")


if __name__ == "__main__":
    main()
