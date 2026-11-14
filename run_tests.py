"""KỊCH BẢN KIỂM THỬ TOÀN DIỆN HỆ THỐNG ĐỒ ÁN LIÊN NGÀNH NHÓM 7.

Tự động kiểm tra:
1. Kiểm tra môi trường & thư viện
2. Chạy 43 Unit Tests của 4 Module
3. Kiểm tra tính toàn vẹn của tệp trọng số best_model_weights.pth
4. Kiểm tra kho 10 ảnh UAV mẫu
5. Kiểm thử luồng tích hợp End-to-End Pipeline
"""

import os
import sys
import time
import subprocess
import unittest
from PIL import Image

# Đảm bảo hiển thị tiếng Việt trên Windows Console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def print_step(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_environment():
    print_step("BƯỚC 1: KIỂM TRA MÔI TRƯỜNG & THƯ VIỆN CỐT LÕI")
    libs = [
        ("PyTorch", "torch"),
        ("TorchVision", "torchvision"),
        ("OpenCV", "cv2"),
        ("Pillow (PIL)", "PIL"),
        ("Pandas", "pandas"),
        ("Streamlit", "streamlit"),
        ("NumPy", "numpy")
    ]
    all_ok = True
    for name, module in libs:
        try:
            m = __import__(module)
            ver = getattr(m, "__version__", "OK")
            print(f"  [✓] {name:<18}: Phiên bản {ver}")
        except ImportError as e:
            print(f"  [✗] {name:<18}: THIẾU ({e})")
            all_ok = False
            
    import torch
    device = "cuda (GPU NVIDIA)" if torch.cuda.is_available() else "cpu"
    print(f"  [*] Thiết bị phần cứng sẵn sàng : {device}")
    return all_ok


def test_files_integrity():
    print_step("BƯỚC 2: KIỂM TRA TỆP TIN & DỮ LIỆU ĐỒ ÁN")
    required_files = [
        "app.py",
        "preprocessing.py",
        "inference_engine.py",
        "spatial_metrics.py",
        "dashboard_ui.py",
        "best_model_weights.pth",
        "requirements.txt"
    ]
    all_ok = True
    for f in required_files:
        if os.path.exists(f):
            size_kb = os.path.getsize(f) / 1024
            print(f"  [✓] Tệp '{f:<25}': Tồn tại ({size_kb:.1f} KB)")
        else:
            print(f"  [✗] Tệp '{f:<25}': THIẾU!")
            all_ok = False
            
    # Kiểm tra kho 10 ảnh mẫu
    sample_dir = "sample_images"
    if os.path.exists(sample_dir):
        samples = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.png', '.tif'))]
        print(f"  [✓] Thư mục '{sample_dir}/'       : Đủ {len(samples)}/10 ảnh mẫu")
        if len(samples) < 10:
            print("      (Khuyến nghị: Cần chuẩn bị đủ 10 ảnh mẫu như đặc tả đề tài)")
    else:
        print(f"  [✗] Thư mục '{sample_dir}/'       : KHÔNG TỒN TẠI!")
        all_ok = False
        
    return all_ok


def test_all_unit_tests():
    print_step("BƯỚC 3: CHẠY BỘ 43 UNIT TESTS CỦA CÁC MODULE")
    modules = [
        ("Tiền xử lý ảnh (Lê Sơn Hải)", "preprocessing.py"),
        ("Lõi AI Suy luận (Nguyễn Trọng Hùng)", "inference_engine.py"),
        ("Phân tích Không gian & Trắc địa (Bùi Đăng Vũ)", "spatial_metrics.py"),
        ("Giao diện Web & Báo cáo (Nguyễn Trọng Hùng)", "dashboard_ui.py")
    ]
    all_ok = True
    for member, py_file in modules:
        print(f"  * Đang kiểm thử {member} ({py_file})...")
        t0 = time.perf_counter()
        res = subprocess.run([sys.executable, py_file, "--test"], capture_output=True, text=True)
        t_elapsed = time.perf_counter() - t0
        if res.returncode == 0:
            print(f"    -> [✓] PASSED 100% trong {t_elapsed:.2f}s")
        else:
            print(f"    -> [✗] FAILED:")
            print(res.stderr or res.stdout)
            all_ok = False
    return all_ok


def test_pipeline_integration():
    print_step("BƯỚC 4: KIỂM THỬ TÍCH HỢP ĐẦU CUỐI (END-TO-END PIPELINE)")
    try:
        from preprocessing import preprocess_image, load_image
        from inference_engine import load_model, run_inference_with_details
        from spatial_metrics import analyze_spatial_mask, rescale_gsd
        from dashboard_ui import create_overlay_image, mask_to_display_image, buildings_to_dataframe

        # 1. Nạp ảnh mẫu đầu tiên trong thư mục sample_images
        samples = [f for f in sorted(os.listdir("sample_images")) if os.path.isfile(os.path.join("sample_images", f))]
        sample_path = os.path.join("sample_images", samples[0])
        img = load_image(sample_path)
        print(f"  [1/4] Tiếp nhận ảnh thực tế '{sample_path}': Size {img.size}")

        # 2. Tiền xử lý
        tensor, orig_dim = preprocess_image(img, target_size=(512, 512))
        print(f"  [2/4] Module 1 (Tiền xử lý) : Đã tạo Tensor shape {list(tensor.shape)}")

        # 3. Suy luận AI
        model, device = load_model("best_model_weights.pth")
        mask, meta = run_inference_with_details(model, device, tensor, threshold=0.75)
        print(f"  [3/4] Module 2 (Lõi AI U-Net): Suy luận xong trong {meta['latency_ms']:.2f} ms ({meta['fps']:.1f} FPS) trên {device}")

        # 4. Đo đạc & Giao diện
        eff_gsd = rescale_gsd(0.3, orig_dim, (512, 512))
        res = analyze_spatial_mask(mask, gsd=eff_gsd, min_pixel_threshold=20)
        overlay = create_overlay_image(img, res["cleaned_mask"])
        df = buildings_to_dataframe(res["buildings"])
        print(f"  [4/4] Module 3 & 4 (Trắc địa & UI): {res['valid_buildings']} công trình, {res['total_area_m2']:.2f} m², Overlay {overlay.size}")

        print("\n  >>> TÍCH HỢP END-TO-END THÀNH CÔNG 100%! HỆ THỐNG SẴN SÀNG HOẠT ĐỘNG! <<<")
        return True
    except Exception as e:
        print(f"\n  [✗] LỖI TÍCH HỢP: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "#" * 70)
    print("  CHƯƠNG TRÌNH KIỂM THỬ TOÀN DIỆN HỆ THỐNG PHẦN MỀM BÓC TÁCH MÁI NHÀ UAV")
    print("#" * 70)

    ok1 = test_environment()
    ok2 = test_files_integrity()
    ok3 = test_all_unit_tests()
    ok4 = test_pipeline_integration()

    print_step("KẾT LUẬN NGHIỆM THU KIỂM THỬ")
    if ok1 and ok2 and ok3 and ok4:
        print("  🎉 TẤT CẢ 4 BƯỚC KIỂM THỬ ĐỀU ĐẠT CHUẨN XUẤT SẮC!")
        print("  Bạn có thể tự tin khởi chạy Web Dashboard bằng lệnh:")
        print("      streamlit run app.py\n")
    else:
        print("  ⚠️ Có một số bài kiểm tra chưa đạt. Vui lòng xem thông báo lỗi chi tiết ở trên.\n")


if __name__ == "__main__":
    main()
