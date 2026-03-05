import os
import tempfile
import gradio as gr
import numpy as np
import pywt
from scipy.fftpack import dct, idct
from PIL import Image
import cv2
from io import BytesIO

# --- CORE UTILITIES WITH FIXED PARAMETERS ---

def generate_fixed_pn_sequence(length: int) -> np.ndarray:
    """Generate a fixed pseudo-noise sequence W of +1/-1 values using numpy seed 42."""
    rng = np.random.RandomState(42)
    return rng.choice([-1.0, 1.0], size=length)

def apply_2level_dwt(Y_channel: np.ndarray):
    """Decompose the luminance channel into 2-level Haar DWT."""
    coeffs1 = pywt.dwt2(Y_channel, 'haar')
    LL1, (LH1, HL1, HH1) = coeffs1
    coeffs2 = pywt.dwt2(LL1, 'haar')
    LL2, (LH2, HL2, HH2) = coeffs2
    return LL2, coeffs1, coeffs2

def reconstruct_from_dwt(LL2_modified, coeffs1, coeffs2):
    """Reconstruct modified Y channel from updated LL2."""
    LL1_orig, (LH1, HL1, HH1) = coeffs1
    LL2_orig, (LH2, HL2, HH2) = coeffs2
    LL1_new = pywt.idwt2((LL2_modified, (LH2, HL2, HH2)), 'haar')
    Y_new = pywt.idwt2((LL1_new, (LH1, HL1, HH1)), 'haar')
    return Y_new

# --- TAB 1: EMBED ---

def embed_watermark(filepath: str):
    if filepath is None:
        raise gr.Error("Please upload an image.")
    
    # Detect original format from extension
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    fmt = 'JPEG' if ext in ['.jpg', '.jpeg'] else 'PNG'
    
    pil_image = Image.open(filepath)
    if pil_image.width < 64 or pil_image.height < 64:
        raise gr.Error("Image must be at least 64x64 pixels.")
    
    # RGB -> YCbCr conversion
    img = pil_image.convert('RGB')
    img_array = np.array(img, dtype=np.float32)
    img_cv = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float64)
    Y, Cr, Cb = img_cv[:,:,0], img_cv[:,:,1], img_cv[:,:,2]
    
    # 2-level Haar DWT
    LL2, coeffs1, coeffs2 = apply_2level_dwt(Y)
    h, w = LL2.shape
    block_size = 8
    N_blocks = (h // block_size) * (w // block_size)
    
    # Generate/tile W to match total blocks
    W = generate_fixed_pn_sequence(1024)
    if len(W) < N_blocks:
        W = np.tile(W, (N_blocks // len(W)) + 1)[:N_blocks]
    
    # Embed W in 8x8 DCT blocks of LL2 (Alpha = 0.5 with adaptive floor)
    LL2_wm = LL2.copy()
    idx = 0
    for row in range(0, h - block_size + 1, block_size):
        for col in range(0, w - block_size + 1, block_size):
            block = LL2_wm[row:row+block_size, col:col+block_size]
            C = dct(dct(block.T, norm='ortho').T, norm='ortho')
            
            # Embedding formula (perceptually adaptive at [3,3] with alpha=0.5 and minimum floor)
            effective_coeff = max(abs(C[3, 3]), 8.0)
            C[3, 3] = C[3, 3] + 0.5 * W[idx % len(W)] * effective_coeff
            
            block_modified = idct(idct(C.T, norm='ortho').T, norm='ortho')
            LL2_wm[row:row+block_size, col:col+block_size] = block_modified
            idx += 1
            
    # Reconstruct Y
    Y_wm = reconstruct_from_dwt(LL2_wm, coeffs1, coeffs2)
    Y_wm = np.clip(Y_wm, 0, 255)
    
    # Merge back
    img_wm = np.stack([Y_wm, Cr, Cb], axis=2).astype(np.uint8)
    img_wm_rgb = cv2.cvtColor(img_wm, cv2.COLOR_YCrCb2RGB)
    result_image = Image.fromarray(img_wm_rgb)
    
    # Save output in same format as input
    buf = BytesIO()
    if fmt == 'JPEG':
        result_image.save(buf, format='JPEG', quality=92)
    else:
        result_image.save(buf, format='PNG')
    buf.seek(0)
    result_image = Image.open(buf).copy()
    
    # Save to temp file with correct extension
    suffix = '.jpg' if fmt == 'JPEG' else '.png'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    if fmt == 'JPEG':
        result_image.save(tmp.name, format='JPEG', quality=92)
    else:
        result_image.save(tmp.name, format='PNG')
    tmp.close()
    
    return result_image, tmp.name

# --- TAB 2: DETECT ---

def detect_watermark(filepath: str):
    if filepath is None:
        raise gr.Error("Please upload an image to analyze.")
        
    suspect_image = Image.open(filepath)
    if suspect_image.width < 64 or suspect_image.height < 64:
        raise gr.Error("Image must be at least 64x64 pixels.")
        
    img = suspect_image.convert('RGB')
    img_array = np.array(img, dtype=np.float32)
    img_cv = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float64)
    Y = img_cv[:,:,0]
    
    # Apply DWT
    LL2, _, _ = apply_2level_dwt(Y)
    h, w = LL2.shape
    block_size = 8
    
    # Extract soft values of C[3,3] clipped to [-50, 50]
    coeff_values = []
    for row in range(0, h - block_size + 1, block_size):
        for col in range(0, w - block_size + 1, block_size):
            block = LL2[row:row+block_size, col:col+block_size].astype(np.float64)
            C = dct(dct(block.T, norm='ortho').T, norm='ortho')
            val = np.clip(C[3, 3], -50.0, 50.0)
            coeff_values.append(val)
            
    extracted_signs = np.array(coeff_values)
    N_actual = len(extracted_signs)
    
    if N_actual == 0:
        return "0.0", "0.0", "ERROR: No blocks analyzed"
        
    # Regenerate W
    W = generate_fixed_pn_sequence(1024)
    if len(W) < N_actual:
        W = np.tile(W, (N_actual // len(W)) + 1)[:N_actual]
        
    # Compute Pearson correlation coefficient rho
    corr_matrix = np.corrcoef(extracted_signs[:N_actual], W[:N_actual])
    rho = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
    threshold = 4.0 / np.sqrt(N_actual)
    
    decision = "DETECTED" if rho > threshold else "NOT DETECTED"
    
    # Debug print as requested
    print(f"DEBUG MATH CORE TEST: rho={rho:.6f}, threshold={threshold:.6f}, N={N_actual}, decision={decision}")
    
    return f"{rho:.6f}", f"{threshold:.6f}", decision

# --- TAB 3: VISUALIZE ---

def analyze_and_visualize(orig_img: Image, wm_img: Image):
    if orig_img is None or wm_img is None:
        raise gr.Error("Please upload both Original and Watermarked images.")
        
    orig_arr = np.array(orig_img.convert('RGB'))
    wm_arr = np.array(wm_img.convert('RGB'))
    
    if orig_arr.shape != wm_arr.shape:
        raise gr.Error("Original and Watermarked images must have the exact same dimensions.")
        
    original_h, original_w, _ = orig_arr.shape
    
    # 1. Difference Map (500x amplified)
    diff = wm_arr.astype(float) - orig_arr.astype(float)
    amplified = np.clip(diff * 500.0 + 128.0, 0, 255).astype(np.uint8)
    
    # 2. Embedding Strength Heatmap
    wm_cv = cv2.cvtColor(wm_arr.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float64)
    Y_wm = wm_cv[:,:,0]
    
    LL2_wm, _, _ = apply_2level_dwt(Y_wm)
    h, w = LL2_wm.shape
    block_size = 8
    block_h = h // block_size
    block_w = w // block_size
    N_blocks = block_h * block_w
    
    heatmap_grid = np.zeros((block_h, block_w), dtype=np.float64)
    coeff_signs = []
    
    r_idx = 0
    for row in range(0, h - block_size + 1, block_size):
        c_idx = 0
        for col in range(0, w - block_size + 1, block_size):
            block = LL2_wm[row:row+block_size, col:col+block_size]
            C = dct(dct(block.T, norm='ortho').T, norm='ortho')
            coeff_mag = abs(C[3, 3])
            heatmap_grid[r_idx, c_idx] = coeff_mag
            
            # Save sign for Vis 3 detection map
            sign = C[3, 3] / coeff_mag if coeff_mag > 1e-10 else 0.0
            coeff_signs.append(sign)
            
            c_idx += 1
        r_idx += 1
        
    max_val = np.max(heatmap_grid)
    if max_val > 1e-10:
        heatmap_norm = (heatmap_grid / max_val * 255).astype(np.uint8)
    else:
        heatmap_norm = np.zeros_like(heatmap_grid, dtype=np.uint8)
        
    heatmap_resized = cv2.resize(heatmap_norm, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    strength_overlay = cv2.addWeighted(orig_arr.astype(np.uint8), 0.5, heatmap_colored_rgb, 0.5, 0)
    
    # 3. Block-level Detection Confidence Map
    extracted_signs = np.array(coeff_signs)
    N_actual = len(extracted_signs)
    
    W = generate_fixed_pn_sequence(1024)
    if len(W) < N_actual:
        W = np.tile(W, (N_actual // len(W)) + 1)[:N_actual]
        
    grid_colors = np.zeros((block_h, block_w, 3), dtype=np.uint8)
    correct_count = 0
    idx = 0
    for r in range(block_h):
        for c in range(block_w):
            if idx < N_actual:
                conf = extracted_signs[idx] * W[idx]
                if conf > 0:
                    grid_colors[r, c] = [0, 255, 0]  # Green
                    correct_count += 1
                else:
                    grid_colors[r, c] = [255, 0, 0]  # Red
            else:
                grid_colors[r, c] = [128, 128, 128]
            idx += 1
            
    percent_correct = (correct_count / N_actual * 100) if N_actual > 0 else 0.0
    confidence_resized = cv2.resize(grid_colors, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    vis3_title_text = f"### Visualization 3 — Block Detection Map — {percent_correct:.1f}% blocks correct"
    
    # 4. Frequency Domain Fingerprint
    green_wm = wm_arr[:,:,1]
    green_orig = orig_arr[:,:,1]
    
    fft_wm = np.fft.fftshift(np.fft.fft2(green_wm))
    log_mag_wm = np.log1p(np.abs(fft_wm))
    
    fft_orig = np.fft.fftshift(np.fft.fft2(green_orig))
    log_mag_orig = np.log1p(np.abs(fft_orig))
    
    fft_diff = np.abs(log_mag_wm - log_mag_orig) * 10.0
    max_fft_diff = np.max(fft_diff)
    if max_fft_diff > 1e-10:
        fft_diff_norm = (fft_diff / max_fft_diff * 255).astype(np.uint8)
    else:
        fft_diff_norm = np.zeros_like(fft_diff, dtype=np.uint8)
        
    fft_colored = cv2.applyColorMap(fft_diff_norm, cv2.COLORMAP_HOT)
    fft_colored_rgb = cv2.cvtColor(fft_colored, cv2.COLOR_BGR2RGB)
    
    # Metrics
    mse = np.mean((orig_arr.astype(float) - wm_arr.astype(float))**2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    
    # Coverage calculation (above 1.0 energy magnitude threshold as baseline signal)
    above_noise = np.sum(heatmap_grid > 1.0)
    coverage_pct = (above_noise / N_blocks * 100) if N_blocks > 0 else 0.0
    mean_strength = np.mean(heatmap_norm)
    
    summary_text = (
        f"### 🔬 Analysis Dashboard Metrics\n"
        f"⚡ **Embedding coverage**: `{coverage_pct:.1f}%` of blocks received watermark signal above noise floor\n\n"
        f"🔥 **Mean embedding strength**: `{mean_strength:.1f}` (scale 0-255)\n\n"
        f"📊 **Estimated PSNR**: `{psnr:.2f} dB` (>42dB = invisible to human eye)"
    )
    
    return amplified, strength_overlay, confidence_resized, fft_colored_rgb, vis3_title_text, summary_text

# --- GRADIO THEMED UI ---

custom_theme = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
).set(
    body_background_fill="*neutral_950",
    body_text_color="*neutral_100",
    block_background_fill="*neutral_900",
    block_border_color="*neutral_800",
    border_color_primary="*neutral_700",
    color_accent_soft="*neutral_800",
    block_title_text_color="*neutral_200",
    input_background_fill="*neutral_800",
)

with gr.Blocks(title="VWE - Pure Math Core Test", theme=custom_theme) as demo:
    gr.Markdown("# VWE — Pure Math Core Test\nStripped testing sandbox to prove out the DWT+DCT math.")
    
    with gr.Tabs():
        with gr.TabItem("🔒 Embed Watermark"):
            with gr.Row():
                img_input = gr.Image(type="filepath", label="Upload Original Image")
                with gr.Column():
                    img_preview = gr.Image(type="pil", label="Preview (display only)")
                    img_download = gr.File(label="Download Watermarked Image (original format)")
            embed_btn = gr.Button("Embed Watermark (Alpha=0.5)", variant="primary")
            embed_btn.click(embed_watermark, inputs=[img_input], outputs=[img_preview, img_download])
            
        with gr.TabItem("🔍 Detect Watermark"):
            with gr.Row():
                suspect_img = gr.Image(type="filepath", label="Upload Suspect Image")
                with gr.Column():
                    rho_val = gr.Textbox(label="Correlation Score ρ")
                    thresh_val = gr.Textbox(label="Detection Threshold (4σ)")
                    decision_val = gr.Textbox(label="Result Decision")
            detect_btn = gr.Button("Detect Watermark", variant="primary")
            detect_btn.click(detect_watermark, inputs=[suspect_img], outputs=[rho_val, thresh_val, decision_val])

        with gr.TabItem("🔬 Visualize Watermark"):
            with gr.Row():
                orig_vis_input = gr.Image(type="pil", label="Original Image")
                wm_vis_input = gr.Image(type="pil", label="Watermarked Image")
            analyze_btn = gr.Button("Analyze & Visualize", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Visualization 1 — Difference Map (500x amplified)")
                    vis1_output = gr.Image(label="Difference Map", show_label=False)
                with gr.Column():
                    gr.Markdown("### Visualization 2 — Embedding Strength Heatmap")
                    vis2_output = gr.Image(label="Embedding Strength", show_label=False)
                    
            with gr.Row():
                with gr.Column():
                    vis3_title = gr.Markdown("### Visualization 3 — Block-level Detection Confidence Map")
                    vis3_output = gr.Image(label="Block Detection Map", show_label=False)
                with gr.Column():
                    gr.Markdown("### Visualization 4 — Frequency Domain Fingerprint")
                    vis4_output = gr.Image(label="FFT Difference Map", show_label=False)
                    
            summary_output = gr.Markdown("### 🔬 Analysis Dashboard Metrics\nUpload images and click **Analyze & Visualize** to see details.")

            analyze_btn.click(
                analyze_and_visualize,
                inputs=[orig_vis_input, wm_vis_input],
                outputs=[vis1_output, vis2_output, vis3_output, vis4_output, vis3_title, summary_output]
            )

if __name__ == "__main__":
    demo.launch()
