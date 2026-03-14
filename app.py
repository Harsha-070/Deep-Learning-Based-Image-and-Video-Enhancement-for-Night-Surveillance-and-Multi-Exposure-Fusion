"""
Streamlit Web Application for Night Surveillance Enhancement System.

Tabs:
    1. Image Enhancement  - Upload and enhance low-light images
    2. Video Enhancement  - Process surveillance video footage
    3. Multi-Exposure Fusion - Handle mixed-exposure images
    4. Object Detection   - Zero-DCE + YOLOv8x detection pipeline
    5. About              - System overview

Usage:
    streamlit run app.py
"""

import os
import io
import tempfile
import streamlit as st

# ─── Lazy heavy imports (only loaded when first needed) ───────────────────────
# torch, cv2, numpy, PIL, models are imported inside cached functions
# so Streamlit starts instantly and loads models on first use.

def _imports():
    """Return commonly used heavy modules, importing once."""
    import numpy as np
    import cv2
    from PIL import Image
    return np, cv2, Image


# ─── Cached Resource Loaders ─────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading enhancement model...")
def load_image_enhancer(model_path):
    from enhance_image import ImageEnhancer
    return ImageEnhancer(model_path=model_path)


@st.cache_resource(show_spinner="Loading video enhancer...")
def load_video_enhancer(model_path, temporal_weight):
    from enhance_video import VideoEnhancer
    return VideoEnhancer(model_path=model_path, temporal_weight=temporal_weight)


@st.cache_resource(show_spinner="Loading fusion engine...")
def load_fuser(model_path):
    from multi_exposure_fusion import MultiExposureFusion
    return MultiExposureFusion(model_path=model_path)


@st.cache_resource(show_spinner="Loading YOLO detector...")
def load_detector(model_path, yolo_model, conf):
    from detect import NightDetector
    return NightDetector(model_path=model_path, yolo_model=yolo_model, conf=conf)


@st.cache_data(show_spinner=False)
def load_checkpoint_info(model_path):
    """Load only epoch/psnr from checkpoint — cached so sidebar doesn't re-read on every click."""
    import torch
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    return ckpt.get("epoch", "N/A"), ckpt.get("psnr", None)


# ─── Main Application ─────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Night Surveillance Enhancement",
        page_icon="🌙",
        layout="wide",
    )

    st.title("Night Surveillance Enhancement System")
    st.markdown("**Zero-DCE Deep Learning Enhancement · YOLOv8x Object Detection · Multi-Exposure Fusion**")

    model_path = "pretrained/best_model.pth"
    model_exists = os.path.exists(model_path)

    # ─── Sidebar ─────────────────────────────────────────────────────────
    st.sidebar.header("System Info")

    if model_exists:
        epoch, psnr = load_checkpoint_info(model_path)
        if isinstance(psnr, float):
            st.sidebar.success(f"Model ready — Epoch {epoch} · PSNR {psnr:.2f} dB")
        else:
            st.sidebar.success("Model ready")
    else:
        st.sidebar.warning("No trained model found.")
        st.sidebar.code("python run_train.py", language="bash")

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
**Quick Start**
1. Train: `python run_train.py`
2. Test: `python test.py --save_visuals`
3. Enhance: upload here
""")

    # ─── Tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📷 Image Enhancement",
        "🎥 Video Enhancement",
        "🔄 Multi-Exposure Fusion",
        "🔍 Object Detection",
        "ℹ️ About",
    ])

    # ─── Tab 1: Image Enhancement ─────────────────────────────────────────
    with tab1:
        st.header("Enhance Low-Light & Night Surveillance Images")
        np, cv2, Image = _imports()

        col_upload, col_settings = st.columns([3, 1])
        with col_upload:
            uploaded_img = st.file_uploader(
                "Upload a low-light image",
                type=["png", "jpg", "jpeg", "bmp"],
                key="img_upload",
            )
        with col_settings:
            post_process = st.checkbox("Post-Processing", value=True,
                                       help="Apply bilateral denoising, CLAHE, and sharpening")
            enhance_btn = st.button("Enhance Image", type="primary", key="img_btn")

        if uploaded_img is not None and enhance_btn:
            if not model_exists:
                st.error("No model weights found. Run `python run_train.py` first.")
            else:
                pil_img = Image.open(uploaded_img).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                with st.spinner("Enhancing..."):
                    enhancer = load_image_enhancer(model_path)
                    enhanced_bgr = enhancer.enhance(img_bgr, post_process=post_process)

                enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(pil_img, caption="Original (Low-Light)", use_container_width=True)
                with col2:
                    st.image(enhanced_rgb, caption="Enhanced", use_container_width=True)

                orig_brightness = np.mean(np.array(pil_img))
                enh_brightness  = np.mean(enhanced_rgb)
                increase = ((enh_brightness - orig_brightness) / max(orig_brightness, 1)) * 100

                m1, m2, m3 = st.columns(3)
                m1.metric("Original Brightness", f"{orig_brightness:.1f}/255")
                m2.metric("Enhanced Brightness", f"{enh_brightness:.1f}/255")
                m3.metric("Brightness Increase",  f"+{increase:.1f}%")

                buf = io.BytesIO()
                Image.fromarray(enhanced_rgb).save(buf, format="PNG")
                st.download_button("Download Enhanced Image", data=buf.getvalue(),
                                   file_name="enhanced_image.png", mime="image/png")

        elif uploaded_img is not None:
            _, _, Image2 = _imports()
            st.image(Image2.open(uploaded_img).convert("RGB"),
                     caption="Uploaded (click Enhance to process)", use_container_width=True)

    # ─── Tab 2: Video Enhancement ─────────────────────────────────────────
    with tab2:
        st.header("Enhance Night Surveillance Video")
        st.info("Temporal EMA smoothing prevents flickering between frames.")

        uploaded_vid = st.file_uploader(
            "Upload a surveillance video",
            type=["mp4", "avi", "mov"],
            key="vid_upload",
        )
        temporal_wt = st.slider("Temporal Smoothing", 0.0, 1.0, 0.85, 0.05,
                                 help="Higher = smoother / less flicker")
        vid_btn = st.button("Enhance Video", type="primary", key="vid_btn")

        if uploaded_vid is not None and vid_btn:
            if not model_exists:
                st.error("No model weights found.")
            else:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_vid.read())
                tfile.close()
                output_path = tfile.name.replace(".mp4", "_enhanced.mp4")

                with st.spinner("Enhancing video (may take a while on CPU)..."):
                    v_enhancer = load_video_enhancer(model_path, temporal_wt)
                    stats = v_enhancer.process_video(tfile.name, output_path,
                                                     show_progress=False)

                st.success("Video enhancement complete!")

                video_bytes = b""
                if os.path.exists(output_path):
                    with open(output_path, "rb") as vf:
                        video_bytes = vf.read()
                    st.video(video_bytes, format="video/mp4")
                else:
                    st.error("Output video not created. Check console for errors.")

                m1, m2, m3 = st.columns(3)
                m1.metric("Frames", stats["total_frames"])
                m2.metric("Processing FPS", f"{stats['avg_fps']:.1f}")
                m3.metric("Total Time", f"{stats['total_time']:.1f}s")

                if video_bytes:
                    st.download_button("Download Enhanced Video", data=video_bytes,
                                       file_name="enhanced_video.mp4", mime="video/mp4")

                for path in [tfile.name, output_path]:
                    try:
                        if os.path.exists(path):
                            os.unlink(path)
                    except OSError:
                        pass

    # ─── Tab 3: Multi-Exposure Fusion ─────────────────────────────────────
    with tab3:
        st.header("Multi-Exposure Fusion")
        st.info("Best for scenes with both very dark and overexposed areas in one image.")
        np2, cv2_2, Image3 = _imports()

        import config as _cfg

        uploaded_mef = st.file_uploader(
            "Upload a mixed-exposure image",
            type=["png", "jpg", "jpeg", "bmp"],
            key="mef_upload",
        )
        method = st.radio("Fusion Method", ["Pyramid Blending", "Mertens Algorithm"],
                          horizontal=True)
        mef_btn = st.button("Apply Fusion", type="primary", key="mef_btn")

        if uploaded_mef is not None and mef_btn:
            pil_img = Image3.open(uploaded_mef).convert("RGB")
            img_bgr = cv2_2.cvtColor(np2.array(pil_img), cv2_2.COLOR_RGB2BGR)

            with st.spinner("Fusing exposures..."):
                fuser = load_fuser(model_path)
                fused_bgr = fuser.fuse(img_bgr) if method == "Pyramid Blending" \
                            else fuser.fuse_opencv_mertens(img_bgr)
                strip_bgr = fuser.create_exposure_strip(img_bgr)

            fused_rgb = cv2_2.cvtColor(fused_bgr, cv2_2.COLOR_BGR2RGB)
            strip_rgb = cv2_2.cvtColor(strip_bgr, cv2_2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            with col1:
                st.image(pil_img, caption="Original", use_container_width=True)
            with col2:
                st.image(fused_rgb, caption="Fused Result", use_container_width=True)

            st.image(strip_rgb, caption=f"Generated Exposures (γ: {_cfg.MEF_EXPOSURES})",
                     use_container_width=True)

            buf = io.BytesIO()
            Image3.fromarray(fused_rgb).save(buf, format="PNG")
            st.download_button("Download Fused Image", data=buf.getvalue(),
                               file_name="fused_image.png", mime="image/png")

    # ─── Tab 4: Object Detection ──────────────────────────────────────────
    with tab4:
        st.header("Night Surveillance Object Detection")
        st.info(
            "Zero-DCE brightens the image first, then **YOLOv8x** detects objects. "
            "Enhancement dramatically improves detection accuracy in dark scenes."
        )
        np3, cv2_3, Image4 = _imports()

        col_up, col_cfg = st.columns([3, 1])
        with col_up:
            det_img = st.file_uploader(
                "Upload a low-light surveillance image",
                type=["png", "jpg", "jpeg", "bmp"],
                key="det_upload",
            )
        with col_cfg:
            yolo_model_choice = st.selectbox(
                "YOLO Model",
                ["yolov8x.pt", "yolov8l.pt", "yolov8m.pt", "yolov8n.pt"],
                index=0,
                help="x = best accuracy · n = fastest",
            )
            conf_thresh = st.slider("Confidence Threshold", 0.10, 0.90, 0.25, 0.05,
                                    help="Lower = more detections")
            enhance_before = st.checkbox("Enhance before detection", value=True,
                                         help="Recommended for dark/night images")
            det_btn = st.button("Detect Objects", type="primary", key="det_btn")

        if det_img is not None and det_btn:
            if not model_exists:
                st.error("No model weights found.")
            else:
                pil_img = Image4.open(det_img).convert("RGB")
                img_bgr = cv2_3.cvtColor(np3.array(pil_img), cv2_3.COLOR_RGB2BGR)

                with st.spinner(f"Enhancing + detecting with {yolo_model_choice}..."):
                    detector = load_detector(model_path, yolo_model_choice, conf_thresh)
                    enhanced_bgr, annotated_bgr, detections = detector.detect(
                        img_bgr, enhance_first=enhance_before
                    )

                enhanced_rgb  = cv2_3.cvtColor(enhanced_bgr,  cv2_3.COLOR_BGR2RGB)
                annotated_rgb = cv2_3.cvtColor(annotated_bgr, cv2_3.COLOR_BGR2RGB)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(pil_img, caption="Original (Dark)", use_container_width=True)
                with c2:
                    st.image(enhanced_rgb, caption="Enhanced (Zero-DCE)", use_container_width=True)
                with c3:
                    st.image(annotated_rgb,
                             caption=f"Detected — {len(detections)} object(s)",
                             use_container_width=True)

                if detections:
                    st.success(f"Found **{len(detections)}** object(s)")
                    st.table([
                        {"Object": d["class"],
                         "Confidence": f"{d['confidence']:.0%}",
                         "Bounding Box": str(d["bbox"])}
                        for d in detections
                    ])
                else:
                    st.warning("No objects detected. Try lowering the confidence threshold.")

                buf = io.BytesIO()
                Image4.fromarray(annotated_rgb).save(buf, format="PNG")
                st.download_button("Download Annotated Image", data=buf.getvalue(),
                                   file_name="detected_objects.png", mime="image/png")

        elif det_img is not None:
            _, _, Image4p = _imports()
            st.image(Image4p.open(det_img).convert("RGB"),
                     caption="Uploaded (click Detect Objects to process)",
                     use_container_width=True)

    # ─── Tab 5: About ─────────────────────────────────────────────────────
    with tab5:
        st.header("System Overview")
        import config as cfg

        st.markdown("""
This system uses **Zero-DCE** for adaptive low-light image enhancement
combined with **YOLOv8x** for object detection in night surveillance footage.

### Enhancement Formula
```
LE(x) = x + α × x × (1 − x)
```
Applied **6 times** per pixel — dark areas are lifted gently while bright areas are protected.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
### Key Features
- Night / Low-Light Image Enhancement
- Video Enhancement with EMA Temporal Smoothing
- Multi-Exposure Fusion (Pyramid + Mertens)
- YOLOv8x Object Detection on Enhanced Frames
- Non-Uniform Illumination Correction
- CLAHE Local Contrast Enhancement
- Detail & Color Preservation
            """)
        with col2:
            st.markdown(f"""
### Model Architecture
- **Enhancement**: Zero-DCE (7-layer CNN, ~79K params)
- **Detection**: YOLOv8x (pretrained on COCO, 80 classes)
- **Curve Iterations**: {cfg.NUM_CURVES}
- **Training Data**: LOL Dataset (485 + 15 pairs)

### Training Losses
- Spatial Consistency · Exposure Control
- Color Constancy · Illumination Smoothness
- L1 Reconstruction · SSIM
            """)

        st.markdown("""
### References
1. Guo et al. "Zero-Reference Deep Curve Estimation." CVPR 2020.
2. Wei et al. "Deep Retinex Decomposition for Low-Light Enhancement." BMVC 2018.
3. Jocher et al. "Ultralytics YOLOv8." 2023.
4. Mertens et al. "Exposure Fusion." Pacific Graphics 2007.
        """)


if __name__ == "__main__":
    main()
