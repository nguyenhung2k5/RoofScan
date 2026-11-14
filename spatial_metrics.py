"""MODULE 3 — BÙI ĐĂNG VŨ: CODE + DEMO + 30 UNIT TEST TRONG MỘT FILE.

Cài: python -m pip install numpy opencv-python-headless
Demo: python spatial_metrics_all_in_one.py
Test: python spatial_metrics_all_in_one.py --test
Mask thực: python spatial_metrics_all_in_one.py --mask mask.png --gsd 0.3
Bật morphology: thêm --morph close_open --kernel-size 3
Kết quả mặc định được xuất vào thư mục module3_results.
Colab: upload file này rồi chạy !python spatial_metrics_all_in_one.py

Nhận mask nhị phân từ Module 2, không tự nhận dạng mái nhà từ ảnh RGB.
GSD của mask thực phải được cung cấp đúng; GSD demo là giả định.
Diện tích chiếu bằng ước tính = số pixel * GSD_x * GSD_y.
Connected Components đếm vùng nối nhau, có thể gộp các mái nhà chạm nhau.
Các hàm có thể import để ghép Module 2 và Dashboard, không tự chạy demo khi import.
"""
from numbers import Integral, Real
from typing import Any, Dict, Tuple, Union

import cv2
import numpy as np

GSD = Union[float, Tuple[float, float]]
__all__ = ["compute_spatial_metrics", "analyze_spatial_mask", "rescale_gsd"]


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} phải là số nguyên.")
    if value < 1:
        raise ValueError(f"{name} phải lớn hơn hoặc bằng 1.")
    return int(value)


def _gsd_axes(gsd: GSD) -> Tuple[float, float]:
    values = gsd if isinstance(gsd, (tuple, list)) else (gsd, gsd)
    if len(values) != 2:
        raise ValueError("GSD hai chiều phải có dạng (gsd_x, gsd_y).")
    axes = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError("gsd phải là số hoặc cặp số dương, đơn vị mét/pixel.")
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("gsd phải hữu hạn và lớn hơn 0.")
        axes.append(value)
    pixel_area = axes[0] * axes[1]
    if not np.isfinite(pixel_area) or pixel_area <= 0:
        raise ValueError("GSD quá lớn hoặc quá nhỏ để tính diện tích bằng float.")
    return axes[0], axes[1]


def _binary_mask(binary_mask: np.ndarray) -> np.ndarray:
    if not isinstance(binary_mask, np.ndarray):
        raise TypeError("binary_mask phải là một mảng NumPy.")
    if binary_mask.ndim != 2 or binary_mask.size == 0:
        raise ValueError("binary_mask phải là mảng 2 chiều không rỗng (H, W).")
    if binary_mask.dtype.kind not in "buif":
        raise TypeError("binary_mask phải có kiểu bool, số nguyên hoặc số thực.")
    if not np.isin(binary_mask, (0, 1, 255)).all():
        raise ValueError("binary_mask chỉ được chứa 0, 1 hoặc 255; không nhận xác suất.")
    return np.ascontiguousarray(binary_mask != 0, dtype=np.uint8)


def rescale_gsd(
    gsd: GSD,
    original_size: Tuple[int, int],
    mask_size: Tuple[int, int],
) -> Tuple[float, float]:
    
    gx, gy = _gsd_axes(gsd)
    sizes = []
    for size, name in ((original_size, "original_size"), (mask_size, "mask_size")):
        if not isinstance(size, (tuple, list)) or len(size) != 2:
            raise ValueError(f"{name} phải có dạng (width, height).")
        sizes.append(tuple(_positive_integer(v, name) for v in size))
    (ow, oh), (mw, mh) = sizes
    return _gsd_axes((gx * ow / mw, gy * oh / mh))


def analyze_spatial_mask(
    binary_mask: np.ndarray,
    gsd: GSD = 0.3,
    min_pixel_threshold: int = 20,
    *,
    morph_operation: str = "none",
    kernel_size: int = 3,
    morph_iterations: int = 1,
    connectivity: int = 8,
) -> Dict[str, Any]:
    
    mask = _binary_mask(binary_mask)
    gx, gy = _gsd_axes(gsd)
    threshold = _positive_integer(min_pixel_threshold, "min_pixel_threshold")
    kernel_size = _positive_integer(kernel_size, "kernel_size")
    iterations = _positive_integer(morph_iterations, "morph_iterations")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size phải là số lẻ.")
    connectivity = _positive_integer(connectivity, "connectivity")
    if connectivity not in (4, 8):
        raise ValueError("connectivity chỉ nhận 4 hoặc 8.")
    operations = {
        "none": (), "open": (cv2.MORPH_OPEN,), "close": (cv2.MORPH_CLOSE,),
        "open_close": (cv2.MORPH_OPEN, cv2.MORPH_CLOSE),
        "close_open": (cv2.MORPH_CLOSE, cv2.MORPH_OPEN),
    }
    if not isinstance(morph_operation, str) or morph_operation not in operations:
        raise ValueError("morph_operation phải là none/open/close/open_close/close_open.")

    if operations[morph_operation] and kernel_size > 1:
        # Nền ngoài ảnh là 0. Padding đủ rộng tránh hiệu ứng biên của chuỗi phép.
        pad = 4 * (kernel_size // 2) * iterations
        padded = np.pad(mask, pad_width=pad, mode="constant")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        for operation in operations[morph_operation]:
            padded = cv2.morphologyEx(
                padded, operation, kernel, iterations=iterations,
                borderType=cv2.BORDER_CONSTANT, borderValue=0,
            )
        mask = np.ascontiguousarray(padded[pad:-pad, pad:-pad])

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=connectivity, ltype=cv2.CV_32S
    )
    remap = np.zeros(num_labels, dtype=np.int32)
    buildings = []
    height, width = mask.shape
    for label_id in range(1, num_labels):
        area_px = int(stats[label_id, cv2.CC_STAT_AREA])
        if area_px < threshold:
            continue
        building_id = len(buildings) + 1
        remap[label_id] = building_id
        x, y, w, h = (int(v) for v in stats[label_id, :4])
        cx, cy = (float(v) for v in centroids[label_id])
        buildings.append({
            "building_id": building_id, "area_pixels": area_px,
            "area_m2": float(area_px * gx * gy),
            "bbox_x": x, "bbox_y": y, "bbox_width": w, "bbox_height": h,
            "centroid_x": cx, "centroid_y": cy,
            "touches_border": x == 0 or y == 0 or x + w == width or y + h == height,
        })

    label_map = remap[labels]
    pixels = sum(b["area_pixels"] for b in buildings)
    return {
        "valid_buildings": len(buildings),
        "total_area_m2": float(pixels * gx * gy),
        "total_building_pixels": pixels,
        "processed_mask": mask,
        "cleaned_mask": (label_map > 0).astype(np.uint8),
        "label_map": label_map, "buildings": buildings,
        "gsd_x": gx, "gsd_y": gy,
    }


def compute_spatial_metrics(
    binary_mask: np.ndarray,
    gsd: GSD = 0.3,
    min_pixel_threshold: int = 20,
    *,
    morph_operation: str = "none",
    kernel_size: int = 3,
    morph_iterations: int = 1,
    connectivity: int = 8,
) -> Tuple[int, float, int]:
    """Giữ giao diện cũ: trả (số vùng, tổng diện tích m², tổng pixel).

    Có thể bật Morphology bằng tham số keyword. Để lấy mask đã lọc,
    ID và số liệu từng vùng cho Dashboard/CSV, dùng analyze_spatial_mask.
    Xem docstring analyze_spatial_mask để biết quy ước và các tham số.
    """
    result = analyze_spatial_mask(
        binary_mask, gsd, min_pixel_threshold,
        morph_operation=morph_operation, kernel_size=kernel_size,
        morph_iterations=morph_iterations, connectivity=connectivity,
    )
    return (
        result["valid_buildings"], result["total_area_m2"],
        result["total_building_pixels"],
    )



# UNIT TEST ĐỘC LẬP — chạy bằng --test
import unittest

import numpy as np



class TestSpatialMetrics(unittest.TestCase):
    """Kiểm tra kết quả đo, quy ước liên thông và dữ liệu đầu vào."""

    def setUp(self):
        # Hai vùng 100 và 50 pixel; một đốm nhiễu 9 pixel.
        self.mask = np.zeros((40, 40), dtype=np.uint8)
        self.mask[2:12, 2:12] = 1
        self.mask[20:25, 20:30] = 1
        self.mask[30:33, 2:5] = 1

    def test_background_returns_zero(self):
        """Ảnh toàn nền không được tính thành một công trình."""
        mask = np.zeros((10, 12), dtype=np.uint8)
        self.assertEqual(compute_spatial_metrics(mask), (0, 0.0, 0))

    def test_all_foreground_is_one_region(self):
        """Toàn bộ 30 pixel công trình là một vùng, diện tích 2,7 m²."""
        count, area, pixels = compute_spatial_metrics(np.ones((5, 6), dtype=np.uint8))
        self.assertEqual((count, pixels), (1, 30))
        self.assertAlmostEqual(area, 2.7)

    def test_default_filters_noise_and_measures_area(self):
        """Giữ 100 + 50 pixel, loại 9 pixel: 2 vùng và 13,5 m²."""
        count, area, pixels = compute_spatial_metrics(self.mask)
        self.assertEqual((count, pixels), (2, 150))
        self.assertAlmostEqual(area, 13.5)

    def test_threshold_boundary_is_inclusive(self):
        """Ngưỡng 20 phải loại vùng 19, giữ vùng 20 và 21 pixel."""
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[1:2, 1:20] = 1  # 19 pixel
        mask[4:8, 1:6] = 1   # 20 pixel
        mask[11:14, 1:8] = 1  # 21 pixel
        count, area, pixels = compute_spatial_metrics(mask, gsd=0.5)
        self.assertEqual((count, pixels), (2, 41))
        self.assertAlmostEqual(area, 10.25)

    def test_threshold_can_remove_every_region(self):
        """Ngưỡng lớn hơn mọi vùng phải trả về toàn số 0."""
        result = compute_spatial_metrics(self.mask, min_pixel_threshold=101)
        self.assertEqual(result, (0, 0.0, 0))

    def test_lower_threshold_keeps_small_region(self):
        """Hạ ngưỡng xuống 1 thì cả vùng 9 pixel cũng được tính."""
        count, area, pixels = compute_spatial_metrics(self.mask, min_pixel_threshold=1)
        self.assertEqual((count, pixels), (3, 159))
        self.assertAlmostEqual(area, 14.31)

    def test_custom_gsd_changes_area_only(self):
        """150 pixel ở GSD 0,5 m/pixel có diện tích 37,5 m²."""
        count, area, pixels = compute_spatial_metrics(self.mask, gsd=0.5)
        self.assertEqual((count, pixels), (2, 150))
        self.assertAlmostEqual(area, 37.5)

    def test_diagonal_pixels_use_eight_connectivity(self):
        """Ba pixel chạm góc là một vùng theo liên thông 8."""
        mask = np.eye(3, dtype=np.uint8)
        count, area, pixels = compute_spatial_metrics(mask, min_pixel_threshold=1)
        self.assertEqual((count, pixels), (1, 3))
        self.assertAlmostEqual(area, 0.27)

    def test_separated_pixels_are_distinct_regions(self):
        """Ba pixel cách nhau bằng nền phải được đếm thành ba vùng."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[0, 0] = mask[2, 2] = mask[4, 4] = 1
        count, area, pixels = compute_spatial_metrics(mask, min_pixel_threshold=1)
        self.assertEqual((count, pixels), (3, 3))
        self.assertAlmostEqual(area, 0.27)

    def test_hole_is_excluded_from_roof_area(self):
        """Vùng 8x8 có lỗ nền 2x2: một vùng và 60 pixel mái nhà."""
        mask = np.ones((8, 8), dtype=np.uint8)
        mask[3:5, 3:5] = 0
        count, area, pixels = compute_spatial_metrics(mask)
        self.assertEqual((count, pixels), (1, 60))
        self.assertAlmostEqual(area, 5.4)

    def test_supported_binary_encodings(self):
        """Mask bool, 0/1 và 0/255 phải cho cùng kết quả đo."""
        variants = {
            "bool": self.mask.astype(bool),
            "0_255": self.mask * 255,
            "float_0_1": self.mask.astype(np.float32),
        }
        for name, mask in variants.items():
            with self.subTest(encoding=name):
                count, area, pixels = compute_spatial_metrics(mask)
                self.assertEqual((count, pixels), (2, 150))
                self.assertAlmostEqual(area, 13.5)

    def test_input_mask_is_not_modified(self):
        """Đo xong phải giữ nguyên cả dữ liệu lẫn kiểu của mask gốc."""
        before = self.mask.copy()
        compute_spatial_metrics(self.mask)
        np.testing.assert_array_equal(self.mask, before)
        self.assertEqual(self.mask.dtype, before.dtype)

    def test_non_contiguous_mask_is_supported(self):
        """Mask lật bằng slicing vẫn đo đúng dù không liên tục bộ nhớ."""
        count, area, pixels = compute_spatial_metrics(self.mask[:, ::-1])
        self.assertEqual((count, pixels), (2, 150))
        self.assertAlmostEqual(area, 13.5)

    def test_invalid_masks_raise_errors(self):
        """Từ chối sai số chiều, mask rỗng, xác suất và nhãn không hợp lệ."""
        cases = [
            ("list", [[0, 1]], TypeError),
            ("one_dimension", np.array([0, 1]), ValueError),
            ("rgb", np.zeros((2, 2, 3)), ValueError),
            ("empty", np.zeros((0, 2)), ValueError),
            ("probability", np.array([[0, 0.8]]), ValueError),
            ("class_2", np.array([[0, 2]]), ValueError),
            ("negative", np.array([[0, -1]]), ValueError),
            ("nan", np.array([[np.nan]]), ValueError),
            ("infinity", np.array([[np.inf]]), ValueError),
            ("text", np.array([["0", "1"]]), TypeError),
        ]
        for name, mask, error_type in cases:
            with self.subTest(case=name):
                with self.assertRaises(error_type):
                    compute_spatial_metrics(mask)

    def test_invalid_gsd_raises_errors(self):
        """GSD phải là số dương hữu hạn, không phải bool hay chuỗi."""
        cases = [
            (0, ValueError), (-0.3, ValueError),
            (np.nan, ValueError), (np.inf, ValueError),
            (True, TypeError), ("0.3", TypeError), (None, TypeError),
        ]
        for value, error_type in cases:
            with self.subTest(gsd=value):
                with self.assertRaises(error_type):
                    compute_spatial_metrics(self.mask, gsd=value)

    def test_invalid_threshold_raises_errors(self):
        """Ngưỡng pixel phải là số nguyên từ 1 trở lên."""
        cases = [
            (0, ValueError), (-1, ValueError), (1.5, TypeError),
            (True, TypeError), ("20", TypeError), (None, TypeError),
        ]
        for value, error_type in cases:
            with self.subTest(threshold=value):
                with self.assertRaises(error_type):
                    compute_spatial_metrics(self.mask, min_pixel_threshold=value)


class TestMorphologyAndSpatialDetails(unittest.TestCase):
    """Kiểm chứng lọc hình thái học, nhãn từng vùng và GSD sau resize."""

    def test_opening_removes_isolated_noise(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[3:8, 3:8] = 1
        mask[14, 14] = 1
        result = analyze_spatial_mask(mask, min_pixel_threshold=1, morph_operation="open")
        expected = np.zeros_like(mask)
        expected[3:8, 3:8] = 1
        np.testing.assert_array_equal(result["cleaned_mask"], expected)
        self.assertEqual(result["valid_buildings"], 1)
        self.assertEqual(result["total_building_pixels"], 25)

    def test_closing_fills_small_hole(self):
        mask = np.zeros((15, 15), dtype=np.uint8)
        mask[3:12, 3:12] = 1
        mask[7, 7] = 0
        result = analyze_spatial_mask(mask, morph_operation="close")
        expected = np.zeros_like(mask)
        expected[3:12, 3:12] = 1
        np.testing.assert_array_equal(result["cleaned_mask"], expected)
        self.assertEqual(result["total_building_pixels"], 81)
        self.assertAlmostEqual(result["total_area_m2"], 7.29)

    def test_both_morphology_orders_handle_noise_and_hole(self):
        mask = np.zeros((25, 25), dtype=np.uint8)
        mask[5:16, 5:16] = 1
        mask[10, 10] = 0
        mask[21, 21] = 1
        for operation in ("open_close", "close_open"):
            with self.subTest(operation=operation):
                result = analyze_spatial_mask(
                    mask, min_pixel_threshold=1, morph_operation=operation
                )
                expected = np.zeros_like(mask)
                expected[5:16, 5:16] = 1
                np.testing.assert_array_equal(result["cleaned_mask"], expected)
                self.assertEqual(result["total_building_pixels"], 121)

    def test_closing_preserves_solid_region_at_image_border(self):
        mask = np.ones((7, 9), dtype=np.uint8)
        result = analyze_spatial_mask(mask, morph_operation="close")
        np.testing.assert_array_equal(result["cleaned_mask"], mask)
        self.assertTrue(result["buildings"][0]["touches_border"])

    def test_morphology_does_not_change_input(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[3:12, 3:12] = 255
        mask[7, 7] = 0
        before = mask.copy()
        analyze_spatial_mask(mask, morph_operation="close_open")
        np.testing.assert_array_equal(mask, before)

    def test_four_connectivity_separates_diagonal_pixels(self):
        result = analyze_spatial_mask(np.eye(3, dtype=np.uint8),
                                      min_pixel_threshold=1, connectivity=4)
        self.assertEqual(result["valid_buildings"], 3)
        self.assertEqual(result["total_building_pixels"], 3)

    def test_cleaned_mask_and_labels_match_reported_pixels(self):
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[1, 1] = 1  # Vùng bị lọc nằm trước hai vùng hợp lệ.
        mask[4:8, 3:8] = 1
        mask[12:16, 20:25] = 1
        result = analyze_spatial_mask(mask)
        self.assertEqual(result["valid_buildings"], 2)
        self.assertEqual(result["total_building_pixels"], 40)
        self.assertEqual(int(result["cleaned_mask"].sum()), 40)
        np.testing.assert_array_equal(np.unique(result["label_map"]), [0, 1, 2])
        self.assertEqual(result["label_map"][1, 1], 0)
        for building in result["buildings"]:
            pixels = np.count_nonzero(result["label_map"] == building["building_id"])
            self.assertEqual(pixels, building["area_pixels"])
        self.assertAlmostEqual(sum(b["area_m2"] for b in result["buildings"]),
                               result["total_area_m2"])

    def test_bbox_and_centroid_use_mask_pixel_coordinates(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[4:8, 3:8] = 1
        building = analyze_spatial_mask(mask)["buildings"][0]
        self.assertEqual((building["bbox_x"], building["bbox_y"],
                          building["bbox_width"], building["bbox_height"]), (3, 4, 5, 4))
        self.assertEqual((building["centroid_x"], building["centroid_y"]), (5.0, 5.5))
        self.assertFalse(building["touches_border"])

    def test_empty_analysis_has_no_phantom_records(self):
        result = analyze_spatial_mask(np.zeros((10, 10), dtype=np.uint8))
        self.assertEqual(result["buildings"], [])
        self.assertEqual(int(result["label_map"].max()), 0)
        self.assertEqual(int(result["cleaned_mask"].sum()), 0)

    def test_rectangular_pixels_have_correct_area(self):
        mask = np.ones((4, 5), dtype=np.uint8)
        count, area, pixels = compute_spatial_metrics(mask, gsd=(0.3, 0.6))
        self.assertEqual((count, pixels), (1, 20))
        self.assertAlmostEqual(area, 3.6)

    def test_resize_gsd_preserves_physical_area(self):
        new_gsd = rescale_gsd(0.3, original_size=(1024, 512), mask_size=(512, 512))
        self.assertEqual(new_gsd, (0.6, 0.3))
        original_area = 1024 * 512 * 0.3 * 0.3
        _, resized_area, _ = compute_spatial_metrics(
            np.ones((512, 512), dtype=np.uint8), gsd=new_gsd
        )
        self.assertAlmostEqual(resized_area, original_area)

    def test_old_api_supports_optional_morphology(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[3:8, 3:8] = 1
        mask[14, 14] = 1
        count, area, pixels = compute_spatial_metrics(
            mask, min_pixel_threshold=1, morph_operation="open"
        )
        self.assertEqual((count, pixels), (1, 25))
        self.assertAlmostEqual(area, 2.25)

    def test_invalid_morphology_parameters(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        cases = [
            ({"morph_operation": "unknown"}, ValueError),
            ({"kernel_size": 2}, ValueError),
            ({"kernel_size": 0}, ValueError),
            ({"kernel_size": 2.5}, TypeError),
            ({"morph_iterations": 0}, ValueError),
            ({"connectivity": 6}, ValueError),
            ({"connectivity": True}, TypeError),
            ({"gsd": (0.3,)}, ValueError),
            ({"gsd": (0.3, -1)}, ValueError),
        ]
        for kwargs, error_type in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(error_type):
                    analyze_spatial_mask(mask, **kwargs)

    def test_invalid_resize_dimensions(self):
        for size in ((0, 512), (512,), (1.5, 512)):
            with self.subTest(size=size):
                with self.assertRaises((TypeError, ValueError)):
                    rescale_gsd(0.3, original_size=size, mask_size=(512, 512))




def export_spatial_results(result, output_dir, *, source, parameters):
    """Xuất PNG, nhãn NPY, CSV và JSON; số liệu dùng cùng mask sau lọc."""
    import csv
    import json
    from pathlib import Path

    parameters = dict(parameters)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, mask in (("input_mask", parameters.pop("input_mask")),
                       ("processed_mask", result["processed_mask"]),
                       ("cleaned_mask", result["cleaned_mask"])):
        # imencode + tofile hỗ trợ đường dẫn Unicode trên Windows.
        ok, encoded = cv2.imencode(".png", _binary_mask(mask) * 255)
        if not ok:
            raise OSError(f"Không thể mã hóa {name}.png")
        encoded.tofile(str(destination / f"{name}.png"))
    np.save(destination / "label_map.npy", result["label_map"])
    fields = ["building_id", "area_pixels", "area_m2", "bbox_x", "bbox_y",
              "bbox_width", "bbox_height", "centroid_x", "centroid_y", "touches_border"]
    with (destination / "buildings.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["buildings"])
    summary = {key: result[key] for key in (
        "valid_buildings", "total_area_m2", "total_building_pixels", "gsd_x", "gsd_y")}
    summary.update(source=source, parameters=parameters)
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination.resolve()


def create_demo_mask():
    """Ba mái nhà, một lỗ nhỏ và các đốm nhiễu; GSD chỉ là giả định."""
    mask = np.zeros((128, 160), dtype=np.uint8)
    mask[12:42, 12:52] = 1
    mask[25:27, 27:29] = 0
    mask[60:90, 90:135] = 1
    mask[100:120, 0:20] = 1
    mask[5:7, 130:132] = 1
    mask[110, 110:115] = 1
    return mask


def main(argv=None):
    """Chạy độc lập: demo, đọc mask thật hoặc chạy unit test."""
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="store_true", help="Chạy 30 unit test độc lập")
    parser.add_argument("--mask", type=Path, help="Mask một kênh 0/1 hoặc 0/255; không phải RGB")
    parser.add_argument("--gsd", type=float, help="GSD mét/pixel; bắt buộc với --mask")
    parser.add_argument("--gsd-y", type=float, help="GSD theo y nếu pixel không vuông")
    parser.add_argument("--min-pixels", type=int, default=20, help="Ngưỡng giữ vùng (mặc định 20)")
    parser.add_argument("--morph", choices=("none", "open", "close", "open_close", "close_open"),
                        help="Mặc định close_open cho demo, none cho mask thật")
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    parser.add_argument("--output", type=Path, default=Path("module3_results"))
    args = parser.parse_args(argv)
    if args.test:
        suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    if args.mask is not None and args.gsd is None:
        parser.error("Phải cung cấp --gsd tương ứng với mask đang đo.")
    if args.gsd_y is not None and args.gsd is None:
        parser.error("--gsd-y cần đi kèm --gsd (GSD theo x).")
    try:
        if args.mask is None:
            mask = create_demo_mask()
            source = "synthetic_demo"
            print("DEMO: mask nhân tạo, GSD giả định 0.3 m/pixel.")
        else:
            # Không convert RGB sang xám hay ngưỡng hóa tự động.
            data = np.fromfile(str(args.mask), dtype=np.uint8)
            mask = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise ValueError("Không thể đọc file mask.")
            source = str(args.mask)
        gsd = args.gsd if args.gsd is not None else 0.3
        if args.gsd_y is not None:
            gsd = (gsd, args.gsd_y)
        operation = args.morph or ("none" if args.mask is not None else "close_open")
        parameters = dict(min_pixel_threshold=args.min_pixels, morph_operation=operation,
                          kernel_size=args.kernel_size, morph_iterations=args.iterations,
                          connectivity=args.connectivity)
        result = analyze_spatial_mask(mask, gsd=gsd, **parameters)
        print(f"Số vùng công trình hợp lệ: {result['valid_buildings']}")
        print(f"Tổng pixel công trình: {result['total_building_pixels']}")
        print(f"Diện tích chiếu bằng ước tính: {result['total_area_m2']:.2f} m²")
        print("ID | Pixel | Diện tích m² | Chạm biên")
        for building in result["buildings"]:
            print(f"{building['building_id']} | {building['area_pixels']} | "
                  f"{building['area_m2']:.2f} | {building['touches_border']}")
        destination = export_spatial_results(result, args.output, source=source,
                                             parameters=dict(parameters, input_mask=mask))
        print(f"Kết quả PNG/NPY/CSV/JSON: {destination}")
    except (ValueError, TypeError, OSError, cv2.error) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
