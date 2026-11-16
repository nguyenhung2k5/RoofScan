"""ỨNG DỤNG HỖ TRỢ BÓC TÁCH VÀ THỐNG KÊ DIỆN TÍCH MÁI NHÀ TỪ ẢNH UAV ĐỘ PHÂN GIẢI CAO.
RoofScan AI - UAV Building Roof Segmentation & Geospatial Metrics.
Đội ngũ phát triển: Nguyễn Trọng Hùng (Lead AI & UI), Bùi Đăng Vũ, Lê Sơn Hải

Khởi chạy ứng dụng:
    streamlit run app.py
"""

from datetime import datetime
import glob
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import streamlit as st
import torch

# Hỗ trợ console hiển thị tiếng Việt
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Import 04 Module chức năng chuyên biệt theo kiến trúc đồ án
try:
    from preprocessing import preprocess_image, load_image
    from inference_engine import load_model, run_inference_with_details
    from spatial_metrics import analyze_spatial_mask, rescale_gsd
    from dashboard_ui import (
        create_overlay_image,
        mask_to_display_image,
        buildings_to_dataframe,
        get_size_distribution,
        get_land_use_metrics,
        convert_dataframe_to_csv,
        convert_image_to_bytes
    )
except ImportError as e:
    st.error(f"Lỗi nạp module đồ án: {e}. Vui lòng đảm bảo các file mã nguồn nằm cùng thư mục.")
    st.stop()


# ==============================================================================
# CẤU HÌNH TRANG WEB & TÙY BIẾN GIAO DIỆN HIỆN ĐẠI (ROOFSCAN THEME)
# ==============================================================================

st.set_page_config(
    page_title="RoofScan AI | Bóc tách & Thống kê Diện tích Mái nhà UAV",
        layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS giao diện tương thích thích ứng 100% cả Light Mode và Dark Mode
st.markdown("""
<style>
    /* 1. Banner tiêu đề trang chủ */
    .main-header {
        background: linear-gradient(135deg, #0d3b66 0%, #005086 50%, #0284c7 100%);
        padding: 22px 24px;
        border-radius: 14px;
        color: #ffffff !important;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(13, 59, 102, 0.25);
    }
    .main-header h1 {
        color: #ffffff !important;
        font-size: 26px;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: 0.3px;
    }
    .main-header p {
        color: #e2e8f0 !important;
        font-size: 14px;
        margin: 0;
    }

    /* 2. Thẻ KPI thích ứng tự nhiên theo theme (Light & Dark) */
    .metric-card {
        background-color: var(--secondary-background-color);
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-left: 5px solid #0284c7;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(2, 132, 199, 0.2);
        border-color: #0284c7;
    }
    .metric-card-title {
        color: var(--text-color);
        opacity: 0.75;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .metric-card-val {
        color: var(--text-color);
        font-size: 28px;
        font-weight: 800;
        margin: 6px 0 2px 0;
    }
    .metric-card-sub {
        color: var(--text-color);
        opacity: 0.55;
        font-size: 12px;
    }

    /* 3. Tiêu đề phân mục thích ứng */
    .section-title {
        color: var(--text-color);
        font-size: 18px;
        font-weight: 700;
        margin: 20px 0 12px 0;
        border-bottom: 2px solid rgba(128, 128, 128, 0.25);
        padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# THANH ĐIỀU KHIỂN BÊN TRÁI (SIDEBAR CONTROLS)
# ==============================================================================

with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0d3b66 0%, #005086 100%); padding: 16px; border-radius: 12px; color: white; text-align: center; margin-bottom: 14px; box-shadow: 0 4px 14px rgba(13, 59, 102, 0.25);">
        <h3 style="margin: 0; color: #ffffff; font-size: 18px; font-weight: 700; letter-spacing: 0.5px;">RoofScan AI</h3>
        <p style="margin: 4px 0 0 0; font-size: 11px; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.8px;">Hệ thống Trắc địa Mái nhà UAV</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Đội ngũ Phát triển Dự án", expanded=False):
        st.markdown("""
        * **Nguyễn Trọng Hùng** (Team Lead): Lõi AI Suy luận & Giao diện Dashboard
        * **Lê Sơn Hải**: Tiếp nhận & Tiền xử lý dữ liệu viễn thám
        * **Bùi Đăng Vũ**: Phân tích Không gian & Đo đạc Trắc địa
        """)

    st.markdown("---")
    st.markdown("#### Cấu hình Tham số Suy luận")

    weights_file = "best_model_weights.pth"
    if not os.path.exists(weights_file):
        st.warning("Chưa tìm thấy tệp trọng số `best_model_weights.pth`.")

    threshold = st.slider(
        "Ngưỡng phân loại (Threshold):",
        min_value=0.10,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Ngưỡng xác suất để nhận diện mái nhà (mặc định 0.50). Bạn có thể tăng lên 0.70 - 0.80 để lọc các bóng râm gắt."
    )

    gsd_value = st.number_input(
        "Độ phân giải mặt đất GSD (m/pixel):",
        min_value=0.01,
        max_value=2.0,
        value=0.30,
        step=0.05,
        format="%.2f",
        help="Ground Sampling Distance: Tỷ lệ mét trên mỗi pixel. Chuẩn WHU Building Dataset là 0.3 m/px."
    )

    min_pixel = st.slider(
        "Lọc đốm nhiễu diện tích (Pixel):",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        help="Loại bỏ các cụm pixel rác nhỏ hơn ngưỡng này (bụi, cây cối, xe cộ)."
    )

    morph_op = st.selectbox(
        "Phép toán hình thái học (Morphology):",
        options=["none", "close_open", "close", "open"],
        index=0,
        help="Làm mịn ranh giới mái nhà, lấp lỗ hổng hoặc tách các mái liền kề."
    )

    with st.expander("Bộ tinh chỉnh ảnh đầu vào (Module 1)", expanded=False):
        st.caption("Cân bằng độ sáng & tương phản cho ảnh thực địa bị mây che hoặc lóa nắng:")
        enhance_brightness = st.slider("Độ sáng (Brightness):", min_value=0.5, max_value=1.5, value=1.0, step=0.05, help="1.0 là mức nguyên bản")
        enhance_contrast = st.slider("Độ tương phản (Contrast):", min_value=0.5, max_value=1.8, value=1.0, step=0.05, help="1.0 là mức nguyên bản")

    overlay_alpha = st.slider(
        "Độ trong suốt lớp phủ (Alpha):",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05
    )

    st.markdown("---")
    st.markdown("#### Tùy biến Hiển thị & Thị giác")

    style_mode = st.selectbox(
        "Kiểu phủ ranh giới:",
        ["Kết hợp cả hai (Viền & Lớp phủ)", "Chỉ vẽ đường viền (Contour)", "Tô đặc diện tích (Solid Fill)"],
        index=0,
        help="Chọn chế độ vẽ viền bao quanh hoặc phủ màu toàn bộ mái nhà"
    )
    style_map = {
        "Kết hợp cả hai (Viền & Lớp phủ)": "both",
        "Chỉ vẽ đường viền (Contour)": "contour",
        "Tô đặc diện tích (Solid Fill)": "solid"
    }

    color_choice = st.selectbox(
        "Bảng màu nhận diện mái:",
        ["Đỏ tươi viễn thám", "Xanh Neon (Cyan)", "Vàng chanh nổi bật", "Xanh lục Emerald"],
        index=0
    )
    color_map = {
        "Đỏ tươi viễn thám": (255, 30, 30),
        "Xanh Neon (Cyan)": (0, 220, 255),
        "Vàng chanh nổi bật": (255, 220, 0),
        "Xanh lục Emerald": (40, 220, 100)
    }

    col_box1, col_box2 = st.columns(2)
    draw_boxes = col_box1.checkbox("Bounding Box", value=True, help="Vẽ khung viền chữ nhật bao quanh từng tòa nhà")
    draw_labels = col_box2.checkbox("Đánh số (#ID)", value=True, help="Ghi nhãn mã số #01, #02... trực tiếp lên mái nhà")


# ==============================================================================
# NẠP MÔ HÌNH HỌC SÂU (MODULE 2 - NGUYỄN TRỌNG HÙNG)
# ==============================================================================

with st.spinner("Đang khởi tạo mạng nơ-ron U-Net và nạp tài nguyên..."):
    model, device = load_model(weights_file)


# ==============================================================================
# GIAO DIỆN TRUNG TÂM (MAIN DASHBOARD)
# ==============================================================================

st.markdown("""
<div class="main-header">
    <h1>Ứng dụng Bóc tách & Thống kê Diện tích Mái nhà từ ảnh UAV</h1>
    <p>Hệ thống hỗ trợ quản lý trắc địa số hóa và đô thị thông minh — Kết hợp Mạng học sâu U-Net và Giải thuật Thị giác không gian.</p>
</div>
""", unsafe_allow_html=True)

# 1. NGUỒN ẢNH ĐẦU VÀO
col_source, col_info = st.columns([2, 1])

with col_source:
    input_mode = st.radio(
        "Chọn phương thức đưa ảnh vào:",
        ["Sử dụng ảnh UAV mẫu (Demo 10 ảnh chuẩn)", "Tải ảnh từ máy tính (PNG, JPG, TIF)"],
        horizontal=True
    )

input_image: Optional[Image.Image] = None
image_name = "uav_image.png"
custom_gt_image: Optional[Image.Image] = None

if input_mode == "Tải ảnh từ máy tính (PNG, JPG, TIF)":
    uploaded_file = st.file_uploader(
        "Chọn tệp ảnh UAV độ phân giải cao:",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="Tải ảnh khảo sát mặt đất bằng drone/UAV"
    )
    if uploaded_file is not None:
        try:
            input_image = load_image(uploaded_file)
            image_name = uploaded_file.name
        except Exception as e:
            st.error(f"Lỗi đọc file ảnh: {e}")

    uploaded_gt = st.file_uploader(
        "Tải kèm tệp nhãn Ground Truth (tùy chọn để tính mIoU):",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="Nếu bạn có mặt nạ nhãn đen-trắng chuẩn của bức ảnh này, hãy tải lên để hệ thống tự động đối chiếu và tính điểm mIoU, Precision, Recall, F1-Score"
    )
    if uploaded_gt is not None:
        try:
            custom_gt_image = load_image(uploaded_gt)
        except Exception as e:
            st.warning(f"Không thể đọc file nhãn Ground Truth: {e}")
else:
    # Lấy danh sách 10 ảnh mẫu hợp lệ trong sample_images
    valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    sample_files = [
        f for f in sorted(glob.glob("sample_images/*.*"))
        if os.path.isfile(f) and f.lower().endswith(valid_exts)
    ]
    if sample_files:
        selected_sample = st.selectbox(
            "Chọn ảnh UAV mẫu trong kho 10 ảnh thử nghiệm:",
            sample_files,
            format_func=lambda x: os.path.basename(x)
        )
        if selected_sample:
            input_image = load_image(selected_sample)
            image_name = os.path.basename(selected_sample)
    else:
        st.warning("Chưa có ảnh mẫu trong thư mục 'sample_images'.")

with col_info:
    if input_image is not None:
        orig_w, orig_h = input_image.size
        extra_note = ""
        if enhance_brightness != 1.0 or enhance_contrast != 1.0:
            extra_note = f"\n* Tiền xử lý: Sáng {enhance_brightness*100:.0f}%, Tương phản {enhance_contrast*100:.0f}%"
        st.info(f"**Thông tin ảnh đang chọn:**\n\n* Tên tệp: `{image_name}`\n* Kích thước: `{orig_w} x {orig_h} px`\n* Thiết bị phần cứng: `{device}`{extra_note}")
    else:
        st.info("Hãy chọn một ảnh mẫu hoặc tải ảnh UAV lên để chạy phân tích.")


# ==============================================================================
# TIẾN TRÌNH THỰC THI (PIPELINE XỬ LÝ 4 MODULES)
# ==============================================================================

if input_image is not None:
    # Áp dụng bộ tinh chỉnh ảnh đầu vào (Module 1 - Tiền xử lý) nếu có thay đổi
    proc_input_image = input_image
    if enhance_brightness != 1.0:
        proc_input_image = ImageEnhance.Brightness(proc_input_image).enhance(enhance_brightness)
    if enhance_contrast != 1.0:
        proc_input_image = ImageEnhance.Contrast(proc_input_image).enhance(enhance_contrast)

    # Nút bấm hành động phục vụ kịch bản demo Bước 3 theo yêu cầu tài liệu
    st.markdown("---")
    btn_col1, btn_col2 = st.columns([1, 2])
    with btn_col1:
        btn_run = st.button(
            "Bóc tách diện tích mái nhà",
            type="primary",
            use_container_width=True,
            help="Bấm nút để kích hoạt mạng AI bóc tách ranh giới và đo đạc diện tích"
        )
    with btn_col2:
        st.caption("*Bấm nút để kích hoạt mạng nơ-ron Deep Learning bóc tách ranh giới mái nhà.*")

    run_key = f"has_run_{image_name}_{enhance_brightness:.2f}_{enhance_contrast:.2f}"
    if btn_run:
        st.session_state[run_key] = True

    if not st.session_state.get(run_key, False):
        st.info("Ảnh đã nạp thành công! Hãy cấu hình tham số suy luận ở thanh bên trái (nếu cần) rồi nhấn nút **'Bóc tách diện tích mái nhà'** ở trên để bắt đầu phân tích.")
        st.image(proc_input_image, caption=f"Ảnh đầu vào đang chờ xử lý: {image_name}", use_container_width=True)
    else:
        st.markdown('<div class="section-title">Kết Quả Bóc Tách & Đo Đạc Trắc Địa</div>', unsafe_allow_html=True)

        # 1. TIỀN XỬ LÝ (MODULE 1 - LÊ SƠN HẢI)
        with st.spinner("Module 1 (Lê Sơn Hải): Đang chuẩn hóa ảnh sang PyTorch Tensor ImageNet..."):
            img_tensor, orig_size = preprocess_image(proc_input_image, target_size=(512, 512))

        # 2. SUY LUẬN HỌC SÂU (MODULE 2 - NGUYỄN TRỌNG HÙNG)
        # Lưu bộ nhớ đệm xác suất per-image để khi kéo slider Threshold/Alpha không phải suy luận lại GPU
        cache_key = f"prob_{image_name}_{weights_file}_{enhance_brightness:.2f}_{enhance_contrast:.2f}"
        if cache_key not in st.session_state or btn_run:
            with st.spinner("Module 2 (Nguyễn Trọng Hùng): Mạng nơ-ron đang tính toán xác suất mái nhà..."):
                _, infer_meta = run_inference_with_details(
                    model=model,
                    device=device,
                    img_tensor=img_tensor,
                    threshold=threshold
                )
                st.session_state[cache_key] = (
                    infer_meta["prob_building"],
                    infer_meta["latency_ms"],
                    infer_meta["fps"]
                )

        prob_building, latency, fps = st.session_state[cache_key]
        binary_mask = (prob_building >= threshold).astype(np.uint8)
        infer_meta = {
            "prob_building": prob_building,
            "latency_ms": latency,
            "fps": fps
        }

        # 3. PHÂN TÍCH KHÔNG GIAN & TRẮC ĐỊA (MODULE 3 - BÙI ĐĂNG VŨ)
        with st.spinner("Module 3 (Bùi Đăng Vũ): Đang chạy Connected Components và tính toán diện tích m²..."):
            # Tự động điều chỉnh tỷ lệ GSD nếu ảnh gốc bị resize về (512, 512)
            effective_gsd = rescale_gsd(
                gsd=gsd_value,
                original_size=orig_size,
                mask_size=(512, 512)
            )
            spatial_results = analyze_spatial_mask(
                binary_mask=binary_mask,
                gsd=effective_gsd,
                min_pixel_threshold=min_pixel,
                morph_operation=morph_op
            )

            # Tính toán chỉ số hình học trắc địa (Chu vi & Độ phức tạp kiến trúc - Chức năng Nâng cấp)
            label_map = spatial_results.get("label_map")
            if label_map is not None:
                for b in spatial_results["buildings"]:
                    b_id = b["building_id"]
                    b_mask = (label_map == b_id).astype(np.uint8)
                    cnts, _ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        peri_px = cv2.arcLength(cnts[0], True)
                    else:
                        peri_px = 2.0 * (b["bbox_width"] + b["bbox_height"])

                    peri_m = round(float(peri_px * effective_gsd[0]), 2)
                    area_px = b["area_pixels"]
                    # Chỉ số Isoperimetric Quotient (Compactness): 4 * pi * Area / (Perimeter^2)
                    compactness = float((4 * np.pi * area_px) / (peri_px ** 2)) if peri_px > 0 else 1.0

                    if compactness >= 0.60:
                        shape_desc = "Chuẩn chữ nhật / Vuông"
                    elif compactness >= 0.35:
                        shape_desc = "Chữ L / Đa giác đều"
                    else:
                        shape_desc = "Phức tạp / Kéo dài"

                    b["perimeter_m"] = peri_m
                    b["compactness"] = round(compactness, 2)
                    b["shape_desc"] = shape_desc

        # 4. GIAO DIỆN & TRÌNH BÀY (NGUYỄN TRỌNG HÙNG)
        valid_count = spatial_results["valid_buildings"]
        total_area = spatial_results["total_area_m2"]
        total_pixels = spatial_results["total_building_pixels"]
        latency = infer_meta["latency_ms"]

        # Hiển thị 4 thẻ KPI chuyên nghiệp
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Số Công Trình Phát Hiện</div>
                <div class="metric-card-val">{valid_count}</div>
                <div class="metric-card-sub">Khối mái nhà riêng biệt</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Tổng Diện Tích Mái Nhà</div>
                <div class="metric-card-val">{total_area:,.2f} <span style="font-size:16px;">m²</span></div>
                <div class="metric-card-sub">GSD hiệu dụng: {effective_gsd[0]:.3f} m/px</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Tổng Pixel Phân Vùng</div>
                <div class="metric-card-val">{total_pixels:,}</div>
                <div class="metric-card-sub">Pixel mái nhà hợp lệ</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Tốc Độ Suy Luận</div>
                <div class="metric-card-val">{latency:.1f} <span style="font-size:16px;">ms</span></div>
                <div class="metric-card-sub">Tương đương {infer_meta['fps']:.1f} FPS ({device})</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # SO SÁNH TRỰC QUAN VÀ TÙY BIẾN THỊ GIÁC
        cleaned_mask = spatial_results["cleaned_mask"]
        overlay_img = create_overlay_image(
            pil_orig=proc_input_image,
            binary_mask=cleaned_mask,
            alpha=overlay_alpha,
            overlay_color=color_map[color_choice],
            style=style_map[style_mode],
            buildings=spatial_results["buildings"],
            draw_boxes=draw_boxes,
            draw_labels=draw_labels
        )
        mask_display_img = mask_to_display_image(cleaned_mask)

        tab_vis1, tab_vis2, tab_vis3 = st.tabs([
            "Ảnh Phủ Ranh Giới (Overlay Blend)",
            "So Sánh Đa Khung Hình (Kèm Ground Truth)",
            "Thống Kê Quy Hoạch & Bảng Chi Tiết"
        ])

        with tab_vis1:
            st.image(
                overlay_img,
                caption=f"Ảnh phủ ranh giới công trình ({color_choice}) - Độ trong suốt Alpha = {overlay_alpha}",
                use_container_width=True
            )

        with tab_vis2:
            gt_mask_path = os.path.join("sample_images", "ground_truth_masks", image_name)
            gt_available = False
            gt_img = None
            gt_title = "4. Nhãn chuẩn Ground Truth"

            if custom_gt_image is not None:
                gt_img = custom_gt_image
                gt_available = True
                gt_title = "4. Nhãn chuẩn Ground Truth (Người dùng tải lên)"
            elif os.path.exists(gt_mask_path):
                try:
                    gt_img = Image.open(gt_mask_path)
                    gt_available = True
                    gt_title = "4. Nhãn chuẩn Ground Truth (WHU)"
                except Exception:
                    gt_available = False

            if gt_available and gt_img is not None:
                try:
                    gt_img_l = gt_img.convert("L")
                    if gt_img_l.size != (cleaned_mask.shape[1], cleaned_mask.shape[0]):
                        gt_img_l = gt_img_l.resize((cleaned_mask.shape[1], cleaned_mask.shape[0]), Image.NEAREST)
                    gt_arr = (np.array(gt_img_l) > 0).astype(np.uint8)
                    pred_arr = (cleaned_mask > 0).astype(np.uint8)

                    inter = int(np.count_nonzero((pred_arr > 0) & (gt_arr > 0)))
                    union = int(np.count_nonzero((pred_arr > 0) | (gt_arr > 0)))
                    sample_iou = (inter / union * 100.0) if union > 0 else 0.0

                    tp = inter
                    fp = int(np.count_nonzero((pred_arr > 0) & (gt_arr == 0)))
                    fn = int(np.count_nonzero((pred_arr == 0) & (gt_arr > 0)))
                    prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
                    rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
                    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

                    st.success(
                        f"**Đối chiếu Ground Truth thực tế**: "
                        f"**mIoU = {sample_iou:.2f}%** | "
                        f"**Precision = {prec:.2f}%** | "
                        f"**Recall = {rec:.2f}%** | "
                        f"**F1-Score = {f1:.2f}%**"
                    )
                except Exception as e:
                    st.warning(f"Không thể đối chiếu nhãn chuẩn: {e}")

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.image(input_image, caption="1. Ảnh chụp UAV gốc", use_container_width=True)
                with c2:
                    st.image(overlay_img, caption="2. Ranh giới mái nhà (Overlay)", use_container_width=True)
                with c3:
                    st.image(mask_display_img, caption="3. Mặt nạ dự đoán (AI Model)", use_container_width=True)
                with c4:
                    try:
                        st.image(gt_img, caption=gt_title, use_container_width=True)
                    except Exception:
                        pass
            else:
                st.info("Chế độ tải ảnh người dùng: Chưa có tệp nhãn Ground Truth tương ứng để tính toán mIoU/F1. (Nếu bạn có tệp nhãn mask đen-trắng của ảnh này, hãy tải kèm ở mục Tải ảnh bên trên để đối chiếu 4 khung hình).")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(input_image, caption="1. Ảnh chụp UAV gốc", use_container_width=True)
                with c2:
                    st.image(overlay_img, caption="2. Ranh giới mái nhà (Overlay)", use_container_width=True)
                with c3:
                    st.image(mask_display_img, caption="3. Mặt nạ nhị phân sạch nhiễu", use_container_width=True)

        with tab_vis3:
            if spatial_results["buildings"]:
                # BIỂU ĐỒ NÂNG CẤP: PHÂN BỐ QUY MÔ & MẬT ĐỘ XÂY DỰNG
                chart_col1, chart_col2 = st.columns([3, 2])
            
                with chart_col1:
                    st.markdown("##### Phân bố quy mô diện tích công trình")
                    df_dist = get_size_distribution(spatial_results["buildings"])
                    try:
                        import altair as alt
                        chart = alt.Chart(df_dist).mark_bar(
                            color="#0284c7",
                            cornerRadiusTopRight=6,
                            cornerRadiusBottomRight=6
                        ).encode(
                            x=alt.X(
                                "Số lượng công trình:Q",
                                title="Số lượng công trình",
                                axis=alt.Axis(tickMinStep=1, labelFontSize=11)
                            ),
                            y=alt.Y(
                                "Phân khúc diện tích:N",
                                sort=None,
                                title=None,
                                axis=alt.Axis(labelAngle=0, labelFontSize=12)
                            ),
                            tooltip=["Phân khúc diện tích", "Số lượng công trình"]
                        ).properties(height=230)

                        text = chart.mark_text(
                            align="left",
                            baseline="middle",
                            dx=6,
                            fontSize=12,
                            fontWeight="bold",
                            color="#0284c7"
                        ).encode(
                            text="Số lượng công trình:Q"
                        )
                        st.altair_chart(chart + text, use_container_width=True, theme="streamlit")
                    except Exception:
                        st.bar_chart(
                            df_dist,
                            x="Số lượng công trình",
                            y="Phân khúc diện tích",
                            color="#0284c7",
                            horizontal=True,
                            height=230
                        )

                with chart_col2:
                    st.markdown("##### Mật độ xây dựng mặt đất")
                    df_land, density_pct = get_land_use_metrics(
                        total_area, orig_size[0], orig_size[1], effective_gsd[0], effective_gsd[1]
                    )
                    total_land_m2 = (orig_size[0] * effective_gsd[0]) * (orig_size[1] * effective_gsd[1])
                    st.metric(
                        label="Tỷ lệ che phủ đất xây dựng (Footprint)",
                        value=f"{density_pct:.1f}%",
                        delta=f"Khảo sát: {total_land_m2:,.0f} m²"
                    )
                    st.dataframe(df_land, hide_index=True, use_container_width=True)

                st.markdown("---")
                st.markdown("##### Soi cận cảnh chi tiết công trình (Building Inspector)")
                insp_col1, insp_col2 = st.columns([1, 2])
                with insp_col1:
                    b_options = [
                        f"#{b['building_id']:02d} — {b['area_m2']:.1f} m² ({b.get('shape_desc', 'Chuẩn')})"
                        for b in spatial_results["buildings"]
                    ]
                    selected_b_idx = st.selectbox(
                        "Chọn công trình để soi cận cảnh:",
                        range(len(b_options)),
                        format_func=lambda idx: b_options[idx]
                    )
                    sel_b = spatial_results["buildings"][selected_b_idx]

                    st.markdown(f"""
                    * **Mã hiệu công trình**: `#{sel_b['building_id']:02d}`
                    * **Diện tích trắc địa**: **`{sel_b['area_m2']:.2f} m²`**
                    * **Chu vi ước tính**: `{sel_b.get('perimeter_m', 0.0)} m`
                    * **Hình thái mái**: `{sel_b.get('shape_desc', 'N/A')}`
                    * **Chỉ số gọn (Compactness)**: `{sel_b.get('compactness', 1.0)}`
                    * **Kích thước hộp bao**: `{sel_b['bbox_width']} x {sel_b['bbox_height']} px`
                    * **Cắt mép ảnh**: `{'Có' if sel_b.get('touches_border') else 'Không'}`
                    """)

                with insp_col2:
                    pad = 25
                    orig_w, orig_h = orig_size
                    mask_h, mask_w = (512, 512)
                    sx = orig_w / mask_w
                    sy = orig_h / mask_h

                    bx = int(sel_b["bbox_x"] * sx)
                    by = int(sel_b["bbox_y"] * sy)
                    bw = int(sel_b["bbox_width"] * sx)
                    bh = int(sel_b["bbox_height"] * sy)

                    x1 = max(0, bx - pad)
                    y1 = max(0, by - pad)
                    x2 = min(orig_w, bx + bw + pad)
                    y2 = min(orig_h, by + bh + pad)

                    crop_orig = proc_input_image.crop((x1, y1, x2, y2))
                    crop_overlay = overlay_img.crop((x1, y1, x2, y2))

                    cr_c1, cr_c2 = st.columns(2)
                    with cr_c1:
                        st.image(crop_orig, caption=f"Ảnh UAV gốc (#{sel_b['building_id']:02d})", use_container_width=True)
                    with cr_c2:
                        st.image(crop_overlay, caption=f"Ranh giới bóc tách AI (#{sel_b['building_id']:02d})", use_container_width=True)

                st.markdown("---")
                st.markdown("##### Danh sách chi tiết từng công trình")

                filter_col1, filter_col2 = st.columns([1, 1])
                with filter_col1:
                    size_filter = st.selectbox(
                        "Lọc công trình theo quy mô:",
                        ["Tất cả công trình", "Chỉ nhà nhỏ (< 50 m²)", "Nhà trung bình (50 - 150 m²)", "Nhà lớn (> 150 m²)"]
                    )

                with filter_col2:
                    sort_order = st.selectbox(
                        "Sắp xếp danh sách:",
                        ["Mặc định (Theo #ID)", "Diện tích giảm dần (Lớn trước)", "Diện tích tăng dần (Nhỏ trước)"]
                    )

                filtered_buildings = list(spatial_results["buildings"])
                if size_filter == "Chỉ nhà nhỏ (< 50 m²)":
                    filtered_buildings = [b for b in filtered_buildings if b.get("area_m2", 0) < 50]
                elif size_filter == "Nhà trung bình (50 - 150 m²)":
                    filtered_buildings = [b for b in filtered_buildings if 50 <= b.get("area_m2", 0) <= 150]
                elif size_filter == "Nhà lớn (> 150 m²)":
                    filtered_buildings = [b for b in filtered_buildings if b.get("area_m2", 0) > 150]

                if sort_order == "Diện tích giảm dần (Lớn trước)":
                    filtered_buildings = sorted(filtered_buildings, key=lambda x: x.get("area_m2", 0.0), reverse=True)
                elif sort_order == "Diện tích tăng dần (Nhỏ trước)":
                    filtered_buildings = sorted(filtered_buildings, key=lambda x: x.get("area_m2", 0.0))

                rows = []
                for b in filtered_buildings:
                    rows.append({
                        "ID Công trình": f"#{b.get('building_id', 0):02d}",
                        "Diện tích (m²)": round(b.get("area_m2", 0.0), 2),
                        "Chu vi (m)": b.get("perimeter_m", 0.0),
                        "Hình thái mái": b.get("shape_desc", "Chuẩn"),
                        "Độ gọn hình học": b.get("compactness", 1.0),
                        "Số Pixel": b.get("area_pixels", 0),
                        "Tọa độ X": b.get("bbox_x", 0),
                        "Tọa độ Y": b.get("bbox_y", 0),
                        "Chiều rộng (px)": b.get("bbox_width", 0),
                        "Chiều cao (px)": b.get("bbox_height", 0),
                        "Chạm biên ảnh": "Có (cắt mép)" if b.get("touches_border", False) else "Không"
                    })
                df_buildings = pd.DataFrame(rows)
                st.dataframe(df_buildings, use_container_width=True, height=280)
            else:
                df_buildings = pd.DataFrame()
                st.warning("Chưa có công trình nào đạt ngưỡng diện tích tối thiểu.")

        # KHU VỰC TẢI VỀ BÁO CÁO (MODULE 4)
        st.markdown('<div class="section-title">Tải Về Kết Quả Báo Cáo Đo Đạc</div>', unsafe_allow_html=True)
        down_col1, down_col2, down_col3, down_col4 = st.columns(4)

        with down_col1:
            mask_bytes = convert_image_to_bytes(mask_display_img, format="PNG")
            st.download_button(
                label="Tải Mặt Nạ Nhị Phân (PNG)",
                data=mask_bytes,
                file_name=f"mask_{os.path.splitext(image_name)[0]}.png",
                mime="image/png",
                use_container_width=True
            )

        with down_col2:
            overlay_bytes = convert_image_to_bytes(overlay_img, format="PNG")
            st.download_button(
                label="Tải Ảnh Phủ Ranh Giới (PNG)",
                data=overlay_bytes,
                file_name=f"overlay_{os.path.splitext(image_name)[0]}.png",
                mime="image/png",
                use_container_width=True
            )

        with down_col3:
            if not df_buildings.empty:
                csv_bytes = convert_dataframe_to_csv(df_buildings)
                st.download_button(
                    label="Tải Thống Kê (CSV Excel)",
                    data=csv_bytes,
                    file_name=f"bao_cao_mai_nha_{os.path.splitext(image_name)[0]}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.button("Tải Thống Kê (CSV)", disabled=True, use_container_width=True)

        with down_col4:
            if spatial_results.get("buildings"):
                now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                max_b = max(spatial_results["buildings"], key=lambda x: x["area_m2"])
                min_b = min(spatial_results["buildings"], key=lambda x: x["area_m2"])
                avg_area = total_area / valid_count if valid_count > 0 else 0.0
                total_land_m2 = (orig_size[0] * effective_gsd[0]) * (orig_size[1] * effective_gsd[1])
                _, density_pct = get_land_use_metrics(
                    total_area, orig_size[0], orig_size[1], effective_gsd[0], effective_gsd[1]
                )

                txt_report = f"""================================================================================
        CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
            Độc lập - Tự do - Hạnh phúc
--------------------------------------------------------------------------------
       BIÊN BẢN NGHIỆM THU KHẢO SÁT HIỆN TRẠNG MÁI NHÀ ĐÔ THỊ BẰNG UAV
                   (Hệ thống Phân tích Tự động RoofScan AI)
================================================================================

1. THÔNG TIN KHẢO SÁT & HỆ THỐNG:
   * Ngày giờ lập biên bản : {now_str}
   * Tệp ảnh khảo sát      : {image_name}
   * Kích thước ảnh gốc    : {orig_size[0]} x {orig_size[1]} px
   * Độ phân giải mặt đất  : {effective_gsd[0]:.3f} m/pixel
   * Thiết bị phần cứng    : {device} (Độ trễ suy luận: {latency:.1f} ms | {infer_meta['fps']:.1f} FPS)
   * Thuật toán AI         : DeepLabV3+ / U-Net (ResNet-50 Backbone)

2. KẾT QUẢ ĐO ĐẠC TRẮC ĐỊA & QUY HOẠCH:
   * Tổng số công trình phát hiện   : {valid_count} khối mái nhà
   * Tổng diện tích mái nhà         : {total_area:,.2f} m²
   * Tổng diện tích khuôn viên đất  : {total_land_m2:,.2f} m²
   * Tỷ lệ che phủ đất xây dựng     : {density_pct:.2f}%
   * Diện tích mái trung bình/căn   : {avg_area:.2f} m²
   * Công trình quy mô lớn nhất     : #{max_b.get('building_id', 'N/A')} ({max_b.get('area_m2', 0):,.2f} m² | Chu vi: {max_b.get('perimeter_m', 0)} m)
   * Công trình quy mô nhỏ nhất     : #{min_b.get('building_id', 'N/A')} ({min_b.get('area_m2', 0):,.2f} m² | Chu vi: {min_b.get('perimeter_m', 0)} m)

3. ĐÁNH GIÁ HIỆU QUẢ KINH TẾ & THỰC THI:
   * Thời gian bóc tách AI          : ~{latency/1000.0:.2f} giây
   * Thời gian đo đạc thủ công ước tính: ~{(valid_count * 25) / 60.0:.1f} giờ làm việc thực địa
   * Mức độ tiết kiệm thời gian     : > 99.9%
   * Tình trạng dữ liệu             : Đã số hóa và chuẩn hóa sẵn sàng nhập hệ thống GIS địa chính.

--------------------------------------------------------------------------------
Biên bản được xuất tự động từ Hệ thống RoofScan AI.
================================================================================
"""
                st.download_button(
                    label="Tải Biên Bản Nghiệm Thu (.TXT)",
                    data=txt_report.encode("utf-8"),
                    file_name=f"bien_ban_nghiem_thu_{os.path.splitext(image_name)[0]}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            else:
                st.button("Tải Biên Bản Nghiệm Thu (.TXT)", disabled=True, use_container_width=True)

else:
    # Trạng thái ban đầu (Empty State)
    st.info("Hãy chọn một ảnh mẫu bên trên hoặc tải ảnh UAV của bạn lên để xem kết quả phân đoạn học sâu.")

st.markdown("---")
st.caption("© 2026 Nhóm Nghiên cứu & Phát triển — Hệ thống Bóc tách & Đo đạc Mái nhà UAV.")
