"""MODULE GIAO DIỆN NGƯỜI DÙNG VÀ XUẤT BÁO CÁO (DASHBOARD UI).
Phụ trách: Nguyễn Trọng Hùng (Trưởng nhóm).

Chức năng:
    - Tạo ảnh phủ ranh giới công trình (Solid Fill, Contour Outline, Bounding Box, Numbering).
    - Tùy biến bảng màu hiển thị viễn thám.
    - Thống kê quy hoạch đô thị: phân bố quy mô diện tích & tỷ lệ che phủ đất (Building Footprint).
    - Hỗ trợ xuất dữ liệu ra file CSV và ảnh PNG mặt nạ phục vụ tải về.

Cách chạy kiểm thử độc lập:
    python dashboard_ui.py --test
"""

import io
import sys
import unittest
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Hỗ trợ console Windows hiển thị tiếng Việt
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def create_overlay_image(
    pil_orig: Image.Image,
    binary_mask: np.ndarray,
    alpha: float = 0.45,
    overlay_color: Tuple[int, int, int] = (255, 30, 30),
    style: str = "solid",
    buildings: Optional[List[dict]] = None,
    draw_boxes: bool = False,
    draw_labels: bool = False,
    box_color: Tuple[int, int, int] = (0, 230, 255)
) -> Image.Image:
    """Tạo ảnh phủ màu bán trong suốt làm nổi bật ranh giới mái nhà trên nền ảnh UAV gốc.
    
    Hỗ trợ 3 phong cách hiển thị:
        - 'solid': Tô đặc toàn bộ diện tích mái nhà bằng màu phủ.
        - 'contour': Chỉ vẽ đường viền ranh giới sắc nét bao quanh mái nhà.
        - 'both': Kết hợp cả tô màu bán trong suốt và vẽ đường viền đậm.
        
    Hỗ trợ vẽ thêm Bounding Box và gắn nhãn số nhà (#01, #02, ...) trực quan.
    """
    if not isinstance(pil_orig, Image.Image):
        raise TypeError(f"pil_orig phải là PIL.Image, nhận được {type(pil_orig)}")
    if not isinstance(binary_mask, np.ndarray):
        raise TypeError(f"binary_mask phải là np.ndarray, nhận được {type(binary_mask)}")

    orig_rgb = pil_orig.convert("RGB")
    orig_w, orig_h = orig_rgb.size

    # Resize mask về kích thước ảnh gốc nếu cần
    mask_h, mask_w = binary_mask.shape[:2]
    if (mask_w, mask_h) != (orig_w, orig_h):
        mask_resized = cv2.resize(
            binary_mask.astype(np.uint8),
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        )
    else:
        mask_resized = binary_mask.astype(np.uint8)

    img_np = np.array(orig_rgb, dtype=np.uint8)
    overlay = img_np.copy()
    is_building = (mask_resized > 0)

    # 1. TÔ MÀU BỀ MẶT (SOLID / BOTH)
    if style in ("solid", "both"):
        overlay[is_building] = overlay_color
        blended = (alpha * overlay + (1.0 - alpha) * img_np).astype(np.uint8)
    else:
        blended = img_np.copy()

    # 2. VẼ ĐƯỜNG VIỀN RANH GIỚI SẮC NÉT (CONTOUR / BOTH)
    if style in ("contour", "both"):
        contours, _ = cv2.findContours(
            mask_resized.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        contour_color = overlay_color if style == "contour" else (255, 255, 255)
        thickness = 2 if orig_w <= 1024 else 3
        cv2.drawContours(blended, contours, -1, contour_color, thickness, cv2.LINE_AA)

    # 3. VẼ BOUNDING BOX VÀ ĐÁNH SỐ THỨ TỰ NHÀ (#01, #02, ...)
    if (draw_boxes or draw_labels) and buildings:
        scale_x = orig_w / mask_w
        scale_y = orig_h / mask_h

        for b in buildings:
            bx = int(b.get("bbox_x", 0) * scale_x)
            by = int(b.get("bbox_y", 0) * scale_y)
            bw = int(b.get("bbox_width", 0) * scale_x)
            bh = int(b.get("bbox_height", 0) * scale_y)
            b_id = b.get("building_id", 0)

            # Vẽ khung chữ nhật
            if draw_boxes:
                cv2.rectangle(blended, (bx, by), (bx + bw, by + bh), box_color, 2, cv2.LINE_AA)

            # Đánh số ID nhà
            if draw_labels:
                label_text = f"#{b_id}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.45 if orig_w <= 1024 else 0.7
                font_thick = 1 if orig_w <= 1024 else 2
                
                # Tính kích thước nền chữ để tạo nhãn đọc rõ
                (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, font_thick)
                bg_x1 = max(0, bx)
                bg_y1 = max(0, by - th - 6)
                bg_x2 = min(orig_w, bg_x1 + tw + 6)
                bg_y2 = min(orig_h, bg_y1 + th + 6)

                cv2.rectangle(blended, (bg_x1, bg_y1), (bg_x2, bg_y2), (15, 23, 42), -1)
                cv2.putText(
                    blended, label_text,
                    (bg_x1 + 3, bg_y2 - 3),
                    font, font_scale, (255, 255, 255),
                    font_thick, cv2.LINE_AA
                )

    return Image.fromarray(blended)


def mask_to_display_image(binary_mask: np.ndarray) -> Image.Image:
    """Chuyển mặt nạ nhị phân (0 và 1) thành ảnh đen trắng chuẩn (0 và 255) để hiển thị."""
    if not isinstance(binary_mask, np.ndarray):
        raise TypeError("binary_mask phải là mảng NumPy.")
    display_arr = (binary_mask > 0).astype(np.uint8) * 255
    return Image.fromarray(display_arr)


def buildings_to_dataframe(buildings: List[dict]) -> pd.DataFrame:
    """Chuyển đổi danh sách thông tin từng công trình sang bảng DataFrame trực quan."""
    if not buildings:
        return pd.DataFrame(columns=[
            "ID Công trình", "Diện tích (m²)", "Số Pixel", "Tọa độ X", "Tọa độ Y", "Chiều rộng", "Chiều cao", "Chạm biên ảnh"
        ])

    rows = []
    for b in buildings:
        rows.append({
            "ID Công trình": f"#{b.get('building_id', 0):02d}",
            "Diện tích (m²)": round(b.get("area_m2", 0.0), 2),
            "Số Pixel": b.get("area_pixels", 0),
            "Tọa độ X": b.get("bbox_x", 0),
            "Tọa độ Y": b.get("bbox_y", 0),
            "Chiều rộng": b.get("bbox_width", 0),
            "Chiều cao": b.get("bbox_height", 0),
            "Chạm biên ảnh": "Có (cắt mép)" if b.get("touches_border", False) else "Không"
        })
    return pd.DataFrame(rows)


def get_size_distribution(buildings: List[dict]) -> pd.DataFrame:
    """Phân nhóm công trình theo các phân khúc quy mô diện tích quy hoạch."""
    categories = {
        "Nhỏ (< 50 m²)": 0,
        "Trung bình (50 - 150 m²)": 0,
        "Lớn (150 - 300 m²)": 0,
        "Rất lớn (> 300 m²)": 0
    }
    for b in buildings:
        area = b.get("area_m2", 0.0)
        if area < 50:
            categories["Nhỏ (< 50 m²)"] += 1
        elif area <= 150:
            categories["Trung bình (50 - 150 m²)"] += 1
        elif area <= 300:
            categories["Lớn (150 - 300 m²)"] += 1
        else:
            categories["Rất lớn (> 300 m²)"] += 1

    df_dist = pd.DataFrame([
        {"Phân khúc diện tích": k, "Số lượng công trình": v} for k, v in categories.items()
    ])
    return df_dist


def get_land_use_metrics(
    total_building_m2: float,
    img_width: int,
    img_height: int,
    gsd_x: float,
    gsd_y: float
) -> Tuple[pd.DataFrame, float]:
    """Tính toán chỉ số mật độ xây dựng (Building Footprint Ratio) trên tổng diện tích đất khảo sát."""
    total_land_m2 = (img_width * gsd_x) * (img_height * gsd_y)
    building_m2 = min(total_building_m2, total_land_m2)
    open_space_m2 = max(0.0, total_land_m2 - building_m2)
    density_pct = (building_m2 / total_land_m2 * 100.0) if total_land_m2 > 0 else 0.0

    df_land = pd.DataFrame([
        {"Loại hình sử dụng đất": "Mái nhà / Đất xây dựng", "Diện tích (m²)": round(building_m2, 2)},
        {"Loại hình sử dụng đất": "Đất trống / Cây xanh / Giao thông", "Diện tích (m²)": round(open_space_m2, 2)}
    ])
    return df_land, density_pct


def convert_dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """Chuyển đổi DataFrame sang định dạng CSV có BOM UTF-8 để mở tiếng Việt trên Microsoft Excel không bị lỗi phông."""
    return df.to_csv(index=False).encode("utf-8-sig")


def convert_image_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    """Chuyển đối tượng ảnh PIL thành chuỗi byte phục vụ nút tải về trên trình duyệt."""
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return buffer.getvalue()


# ==============================================================================
# BỘ KIỂM THỬ ĐỘC LẬP (UNIT TESTS) — Chạy bằng: python dashboard_ui.py --test
# ==============================================================================

class TestDashboardUI(unittest.TestCase):
    """Kiểm thử unit test độc lập cho Dashboard UI."""

    def test_create_overlay_image_dimensions(self):
        """Ảnh overlay trả về phải giữ nguyên kích thước của ảnh gốc."""
        orig = Image.new("RGB", (640, 480), color=(100, 100, 100))
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:200, 100:200] = 1

        overlay_img = create_overlay_image(orig, mask, alpha=0.5)
        self.assertEqual(overlay_img.size, (640, 480))
        self.assertEqual(overlay_img.mode, "RGB")

    def test_contour_and_box_drawing(self):
        """Kiểm tra chức năng vẽ đường viền contour và gắn nhãn số nhà."""
        orig = Image.new("RGB", (512, 512), color=(100, 100, 100))
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[50:150, 50:150] = 1
        buildings = [{
            "building_id": 1, "area_m2": 900.0, "area_pixels": 10000,
            "bbox_x": 50, "bbox_y": 50, "bbox_width": 100, "bbox_height": 100
        }]
        img = create_overlay_image(orig, mask, style="both", buildings=buildings, draw_boxes=True, draw_labels=True)
        self.assertEqual(img.size, (512, 512))

    def test_size_distribution_and_land_use(self):
        """Kiểm tra tính toán phân bố diện tích và mật độ xây dựng."""
        buildings = [
            {"building_id": 1, "area_m2": 40.0},
            {"building_id": 2, "area_m2": 120.0},
            {"building_id": 3, "area_m2": 250.0},
            {"building_id": 4, "area_m2": 450.0}
        ]
        df_dist = get_size_distribution(buildings)
        self.assertEqual(len(df_dist), 4)
        self.assertEqual(df_dist["Số lượng công trình"].sum(), 4)

        df_land, density = get_land_use_metrics(860.0, 512, 512, 0.3, 0.3)
        self.assertGreater(density, 0.0)
        self.assertEqual(len(df_land), 2)


def main():
    if "--test" in sys.argv:
        print("[Dashboard UI] Đang chạy kiểm thử Unit Test...")
        sys.argv.remove("--test")
        unittest.main()
    else:
        print("=" * 60)
        print("MODULE GIAO DIỆN VÀ XUẤT BÁO CÁO (DASHBOARD UI)")
        print("Phụ trách: Nguyễn Trọng Hùng (Trưởng nhóm)")
        print("=" * 60)
        dummy_img = Image.new("RGB", (512, 512), color=(80, 120, 80))
        dummy_mask = np.zeros((512, 512), dtype=np.uint8)
        dummy_mask[100:250, 100:250] = 1
        overlay = create_overlay_image(dummy_img, dummy_mask)
        print(f"Tạo ảnh overlay thành công: Size={overlay.size}")
        print("Chạy 'python dashboard_ui.py --test' để kiểm thử độc lập.")


if __name__ == "__main__":
    main()
