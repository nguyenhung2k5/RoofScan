# 🛰️ RoofScan AI: UAV Building Roof Segmentation & Geospatial Metrics

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-ff4b4b.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-WHU%20Building-success.svg)](http://gpcv.whu.edu.cn/data/building_dataset.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**RoofScan AI** là hệ thống Trí tuệ Nhân tạo hỗ trợ tự động hóa bóc tách ranh giới và đo đạc diện tích mái nhà thực tế ($m^2$) từ ảnh chụp thiết bị bay không người lái (UAV/Drone) độ phân giải cao, phục vụ công tác trắc địa, quản lý đất đai và quy hoạch đô thị thông minh.

---

## 🌟 Tính Năng Nổi Bật

- **Lõi AI Phân Đoạn Ngữ Nghĩa Vượt Trội**: Ứng dụng kiến trúc **DeepLabV3+ (backbone ResNet-50 + ASPP)** được huấn luyện trên tập dữ liệu chuẩn viễn thám **WHU Aerial Building Dataset**, đạt **Validation mIoU: 89.02%** và **F1-Score: 88.62%**.
- **Đo Đạc Trắc Địa Tự Động (Geospatial Metrics)**: Quy đổi chính xác từ số lượng pixel sang diện tích thực tế mặt đất ($m^2$) thông qua hệ số **GSD (Ground Sampling Distance)**, tự động điều chỉnh tỷ lệ đa độ phân giải.
- **Phát Hiện & Đánh Số Công Trình (#ID Bounding Box)**: Tự động tách từng khối nhà riêng biệt bằng giải thuật *Connected Components*, bao khung Bounding Box và gắn nhãn mã định danh `#01, #02...` trực tiếp trên từng nóc nhà.
- **Tùy Biến Thị Giác Đa Dạng**: Hỗ trợ 3 chế độ hiển thị (*Chỉ vẽ viền Contour sắc nét*, *Tô đặc diện tích*, hoặc *Kết hợp cả hai*) với 4 bảng màu viễn thám chuyên dụng (Đỏ tươi, Xanh Cyan Neon, Vàng chanh, Xanh lục Emerald).
- **Đối Chiếu Nhãn Chuẩn Trực Quan (Ground Truth Validation)**: So sánh đồng thời 4 khung hình song song và tính toán tự động các chỉ số học thuật thời gian thực: **mIoU**, **Precision**, **Recall**, **F1-Score**.
- **Phân Tích Quy Hoạch Đô Thị (Urban Analytics)**: Biểu đồ thanh ngang phân loại công trình theo 4 phân khúc quy mô diện tích và tính toán tỷ lệ che phủ đất xây dựng (*Building Footprint Ratio*).
- **Xuất Báo Cáo Chuẩn Hóa**: Tải về tệp báo cáo **CSV (chuẩn UTF-8 BOM)** xem ngay trên Microsoft Excel không bị lỗi phông tiếng Việt và ảnh mặt nạ PNG độ phân giải cao.

---

## 🏗️ Kiến Trúc Hệ Thống (4 Modules Pipeline)

```
[Ảnh UAV Gốc]
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ Module 1: Tiếp nhận & Tiền xử lý (Lê Sơn Hải)          │
│ • Chuẩn hóa kích thước (512x512) & Tensor PyTorch      │
│ • Chuẩn hóa dải màu viễn thám theo phân phối ImageNet  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Module 2: Lõi AI Suy luận Học sâu (Nguyễn Trọng Hùng)  │
│ • DeepLabV3+ (ResNet-50 Backbone) trên GPU CUDA        │
│ • Tính toán bản đồ xác suất (Probability Map)          │
│ • Phân ngưỡng động (Dynamic Thresholding)              │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Module 3: Đo đạc Không gian & Trắc địa (Bùi Đăng Vũ)   │
│ • Connected Components Labeling phân cụm công trình    │
│ • Lọc nhiễu hình thái học (Morphological Operations)   │
│ • Tính diện tích thực tế (m²) theo GSD Rescale         │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Module 4: Giao diện Web & Báo cáo (Nguyễn Trọng Hùng)  │
│ • Bảng điều khiển tương tác Streamlit Web Dashboard    │
│ • Lớp phủ đa màu sắc, Contour, Bounding Box & #ID tags │
│ • Biểu đồ quy hoạch, đối soát Ground Truth & Xuất CSV  │
└────────────────────────────────────────────────────────┘
```

---

## 👥 Đội Ngũ Phát Triển Dự Án

* **Nguyễn Trọng Hùng** (Team Lead / Lead AI & UI):
  * Thiết kế kiến trúc tổng thể dự án.
  * Phụ trách Module 2 (Lõi AI Suy luận Học sâu, DeepLabV3+, Thresholding, GPU Caching).
  * Phụ trách Module 4 (Giao diện Dashboard Streamlit, Tùy biến thị giác, Phân tích quy hoạch & Xuất báo cáo).
* **Bùi Đăng Vũ** (Spatial Metrics Engineer):
  * Phụ trách Module 3 (Thuật toán Connected Components, Lọc nhiễu hình thái học, Tính toán diện tích trắc địa theo GSD).
* **Lê Sơn Hải** (Data Preprocessing Engineer):
  * Phụ trách Module 1 (Pipeline tiếp nhận dữ liệu viễn thám, Đọc ảnh đa định dạng, Resize chuẩn hóa ImageNet).

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy

### 1. Yêu Cầu Môi Trường
- Hệ điều hành: Windows / Linux / macOS
- Python 3.10 trở lên
- Card đồ họa NVIDIA hỗ trợ CUDA (khuyến nghị, hệ thống có cơ chế tự động chuyển sang CPU nếu không có GPU).

### 2. Cài Đặt Thư Viện
```bash
# Clone dự án từ GitHub
git clone https://github.com/<tai-khoan-cua-ban>/RoofScan-AI.git
cd RoofScan-AI

# Cài đặt các gói phụ thuộc
pip install -r requirements.txt
```

### 3. Tải Trọng Số Mô Hình
Đặt tệp trọng số đã huấn luyện `best_model_weights.pth` (khoảng 484 MB) vào ngay thư mục gốc của dự án `RoofScan-AI/`.

### 4. Khởi Chạy Ứng Dụng Web
```bash
streamlit run app.py
```
Sau khi khởi chạy, mở trình duyệt web tại địa chỉ: `http://localhost:8501`.

---

## 🧪 Kiểm Thử Tự Động (Automated Testing)

Dự án tích hợp sẵn bộ kiểm thử độc lập gồm **43 bài Unit Tests** và bài test tích hợp **End-to-End Pipeline**:

```bash
# Chạy bộ kiểm thử toàn diện
python run_tests.py
```

Hoặc chạy kiểm thử riêng lẻ từng module:
```bash
python preprocessing.py --test      # Kiểm thử Module 1
python inference_engine.py --test   # Kiểm thử Module 2
python spatial_metrics.py --test    # Kiểm thử Module 3
python dashboard_ui.py --test       # Kiểm thử Module 4
```

---

## 📄 Bản Quyền & Giấy Phép
Dự án được phân phối dưới giấy phép **MIT License**. Mọi đóng góp và mã nguồn mở đều được hoan nghênh.
