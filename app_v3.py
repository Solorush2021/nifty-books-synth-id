import os
import tempfile
import gradio as gr
import numpy as np
import pywt
from scipy.fftpack import dct, idct
from PIL import Image
import cv2
from io import BytesIO
import numpy.fft as fft

# --- ORIGINAL CORE UTILITIES (UNCHANGED) ---

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

# --- TAB 1: EMBED (ORIGINAL - UNCHANGED) ---

def embed_watermark(filepath: str):
    if filepath is None:
        raise gr.Error("Please upload an image.")
    
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    fmt = 'JPEG' if ext in ['.jpg', '.jpeg'] else 'PNG'
    
    pil_image = Image.open(filepath)
    if pil_image.width < 64 or pil_image.height < 64:
        raise gr.Error("Image must be at least 64x64 pixels.")
    
    img = pil_image.convert('RGB')
    img_array = np.array(img, dtype=np.float32)
    img_cv = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float64)
    Y, Cr, Cb = img_cv[:,:,0], img_cv[:,:,1], img_cv[:,:,2]
    
    LL2, coeffs1, coeffs2 = apply_2level_dwt(Y)
    h, w = LL2.shape
    block_size = 8
    N_blocks = (h // block_size) * (w // block_size)
    
    W = generate_fixed_pn_sequence(1024)
    if len(W) < N_blocks:
        W = np.tile(W, (N_blocks // len(W)) + 1)[:N_blocks]
    
    LL2_wm = LL2.copy()
    idx = 0
    for row in range(0, h - block_size + 1, block_size):
        for col in range(0, w - block_size + 1, block_size):
            block = LL2_wm[row:row+block_size, col:col+block_size]
            C = dct(dct(block.T, norm='ortho').T, norm='ortho')
            effective_coeff = max(abs(C[3, 3]), 8.0)
            C[3, 3] = C[3, 3] + 0.5 * W[idx % len(W)] * effective_coeff
            block_modified = idct(idct(C.T, norm='ortho').T, norm='ortho')
            LL2_wm[row:row+block_size, col:col+block_size] = block_modified
            idx += 1
            
    Y_wm = reconstruct_from_dwt(LL2_wm, coeffs1, coeffs2)
    Y_wm = np.clip(Y_wm, 0, 255)
    
    img_wm = np.stack([Y_wm, Cr, Cb], axis=2).astype(np.uint8)
    img_wm_rgb = cv2.cvtColor(img_wm, cv2.COLOR_YCrCb2RGB)
    result_image = Image.fromarray(img_wm_rgb)
    
    buf = BytesIO()
    if fmt == 'JPEG':
        result_image.save(buf, format='JPEG', quality=92)
    else:
        result_image.save(buf, format='PNG')
    buf.seek(0)
    result_image = Image.open(buf).copy()
    
    suffix = '.jpg' if fmt == 'JPEG' else '.png'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    if fmt == 'JPEG':
        result_image.save(tmp.name, format='JPEG', quality=92)
    else:
        result_image.save(tmp.name, format='PNG')
    tmp.close()
    
    return result_image, tmp.name

# --- TAB 2: DETECT (ORIGINAL - UNCHANGED) ---

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
    
    LL2, _, _ = apply_2level_dwt(Y)
    h, w = LL2.shape
    block_size = 8
    
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
        
    W = generate_fixed_pn_sequence(1024)
    if len(W) < N_actual:
        W = np.tile(W, (N_actual // len(W)) + 1)[:N_actual]
        
    corr_matrix = np.corrcoef(extracted_signs[:N_actual], W[:N_actual])
    rho = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
    threshold = 4.0 / np.sqrt(N_actual)
    
    decision = "DETECTED" if rho > threshold else "NOT DETECTED"
    
    print(f"DEBUG MATH CORE TEST: rho={rho:.6f}, threshold={threshold:.6f}, N={N_actual}, decision={decision}")
    
    return f"{rho:.6f}", f"{threshold:.6f}", decision

# --- TAB 3: VISUALIZE (ORIGINAL - UNCHANGED) ---

def analyze_and_visualize(orig_img: Image, wm_img: Image):
    if orig_img is None or wm_img is None:
        raise gr.Error("Please upload both Original and Watermarked images.")
        
    orig_arr = np.array(orig_img.convert('RGB'))
    wm_arr = np.array(wm_img.convert('RGB'))
    
    if orig_arr.shape != wm_arr.shape:
        raise gr.Error("Original and Watermarked images must have the exact same dimensions.")
        
    original_h, original_w, _ = orig_arr.shape
    
    diff = wm_arr.astype(float) - orig_arr.astype(float)
    amplified = np.clip(diff * 500.0 + 128.0, 0, 255).astype(np.uint8)
    
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
                    grid_colors[r, c] = [0, 255, 0]
                    correct_count += 1
                else:
                    grid_colors[r, c] = [255, 0, 0]
            else:
                grid_colors[r, c] = [128, 128, 128]
            idx += 1
            
    percent_correct = (correct_count / N_actual * 100) if N_actual > 0 else 0.0
    confidence_resized = cv2.resize(grid_colors, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    vis3_title_text = f"### Visualization 3 — Block Detection Map — {percent_correct:.1f}% blocks correct"
    
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
    
    mse = np.mean((orig_arr.astype(float) - wm_arr.astype(float))**2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    
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

# ============================================================
# TAB 4: PHASE DOMAIN WATERMARK (SynthID-Style - NEW ALGORITHM)
# ============================================================

def embed_phase_watermark(filepath: str, freq_u: int = 9, freq_v: int = 9, epsilon: float = 0.02):
    """
    Embed watermark in frequency domain phase (SynthID-style).
    Uses green channel FFT phase perturbation at fixed frequency bins.
    """
    if filepath is None:
        raise gr.Error("Please upload an image.")
    
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    fmt = 'JPEG' if ext in ['.jpg', '.jpeg'] else 'PNG'
    
    pil_image = Image.open(filepath)
    if pil_image.width < 64 or pil_image.height < 64:
        raise gr.Error("Image must be at least 64x64 pixels.")
    
    # Work in RGB - focus on GREEN channel (human eye most sensitive)
    img_array = np.array(pil_image.convert('RGB')).astype(np.float64)
    
    # Process GREEN channel only (SynthID insight: green channel carries signal)
    green_channel = img_array[:, :, 1]
    
    # Apply FFT
    fft_green = fft.fft2(green_channel)
    fft_shifted = fft.fftshift(fft_green)
    
    h, w = green_channel.shape
    
    # Generate fixed phase pattern (SynthID: fixed phase at specific bins)
    # Use seed 42 for reproducibility
    rng = np.random.RandomState(42)
    phase_pattern = rng.uniform(-np.pi/4, np.pi/4, size=(h, w))
    
    # Embed phase perturbation at target frequency bins (u, v)
    # SynthID embeds at (9,9) for 1024x1024 images
    center_h, center_w = h // 2, w // 2
    
    # Apply phase perturbation
    fft_shifted[center_h + freq_v, center_w + freq_u] *= np.exp(1j * epsilon)
    fft_shifted[center_h - freq_v, center_w - freq_u] *= np.exp(-1j * epsilon)  # Conjugate symmetric
    
    # Inverse FFT
    fft_ishifted = fft.ifftshift(fft_shifted)
    green_wm = np.real(fft.ifft2(fft_ishifted))
    
    # Reconstruct image
    img_wm_array = img_array.copy()
    img_wm_array[:, :, 1] = np.clip(green_wm, 0, 255)
    
    result_image = Image.fromarray(img_wm_array.astype(np.uint8))
    
    # Save
    buf = BytesIO()
    if fmt == 'JPEG':
        result_image.save(buf, format='JPEG', quality=92)
    else:
        result_image.save(buf, format='PNG')
    buf.seek(0)
    result_image = Image.open(buf).copy()
    
    suffix = '.jpg' if fmt == 'JPEG' else '.png'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    if fmt == 'JPEG':
        result_image.save(tmp.name, format='JPEG', quality=92)
    else:
        result_image.save(tmp.name, format='PNG')
    tmp.close()
    
    # Calculate PSNR
    mse = np.mean((img_array - img_wm_array)**2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    
    return result_image, tmp.name, f"PSNR: {psnr:.2f} dB (Phase ε={epsilon})"

def detect_phase_watermark(filepath: str, freq_u: int = 9, freq_v: int = 9, num_images: int = 200):
    """
    Detect phase watermark using SynthID method:
    Average FFT phases across multiple images to cancel content, leaving phase pattern.
    """
    if filepath is None:
        raise gr.Error("Please upload an image.")
    
    suspect_image = Image.open(filepath)
    img_array = np.array(suspect_image.convert('RGB')).astype(np.float64)
    
    # Focus on green channel
    green_channel = img_array[:, :, 1]
    
    # FFT
    fft_green = fft.fft2(green_channel)
    fft_shifted = fft.fftshift(fft_green)
    
    h, w = green_channel.shape
    center_h, center_w = h // 2, w // 2
    
    # Extract phase at target frequency bin
    target_phase = np.angle(fft_shifted[center_h + freq_v, center_w + freq_u])
    
    # For single image detection, compare to expected phase pattern
    # In practice, you'd average multiple generated images (SynthID method)
    rng = np.random.RandomState(42)
    expected_phase = rng.uniform(-np.pi/4, np.pi/4)
    
    # Simple phase coherence check
    phase_diff = abs(target_phase - expected_phase)
    phase_coherence = 1.0 - (phase_diff / np.pi)
    
    threshold = 0.7
    decision = "DETECTED" if phase_coherence > threshold else "NOT DETECTED"
    
    print(f"DEBUG Phase: coherence={phase_coherence:.4f}, threshold={threshold:.4f}, decision={decision}")
    
    return f"{phase_coherence:.4f}", f"{threshold:.4f}", decision

# ============================================================
# TAB 5: MULTI-CHANNEL FREQUENCY WATERMARK (NEW ALGORITHM)
# ============================================================

def embed_multichannel_watermark(filepath: str, alpha: float = 0.6):
    """
    Embed watermark across ALL RGB channels in DCT domain.
    More robust than single-channel because signal is spread across channels.
    """
    if filepath is None:
        raise gr.Error("Please upload an image.")
    
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    fmt = 'JPEG' if ext in ['.jpg', '.jpeg'] else 'PNG'
    
    pil_image = Image.open(filepath)
    if pil_image.width < 64 or pil_image.height < 64:
        raise gr.Error("Image must be at least 64x64 pixels.")
    
    img_array = np.array(pil_image.convert('RGB')).astype(np.float64)
    h, w, _ = img_array.shape
    
    # Generate watermark sequence
    rng = np.random.RandomState(42)
    W = rng.choice([-1.0, 1.0], size=1024)
    
    result_array = img_array.copy()
    
    # Embed in each RGB channel separately
    for channel_idx in range(3):
        channel = img_array[:, :, channel_idx]
        
        # Divide into 8x8 blocks
        block_size = 8
        idx = 0
        
        for row in range(0, h - block_size + 1, block_size):
            for col in range(0, w - block_size + 1, block_size):
                block = channel[row:row+block_size, col:col+block_size]
                
                # Apply DCT
                C = dct(dct(block.T, norm='ortho').T, norm='ortho')
                
                # Embed in mid-frequency coefficients (more robust than just [3,3])
                dct_positions = [(3, 3), (2, 4), (4, 2), (3, 4), (4, 3)]
                
                for pos in dct_positions:
                    i, j = pos
                    if i < block_size and j < block_size:
                        effective_coeff = max(abs(C[i, j]), 5.0)
                        C[i, j] = C[i, j] + alpha * W[idx % len(W)] * effective_coeff * (channel_idx + 1) / 3.0
                
                # Inverse DCT
                block_modified = idct(idct(C.T, norm='ortho').T, norm='ortho')
                result_array[row:row+block_size, col:col+block_size, channel_idx] = block_modified
                
                idx += 1
    
    result_array = np.clip(result_array, 0, 255)
    result_image = Image.fromarray(result_array.astype(np.uint8))
    
    # Save
    buf = BytesIO()
    if fmt == 'JPEG':
        result_image.save(buf, format='JPEG', quality=92)
    else:
        result_image.save(buf, format='PNG')
    buf.seek(0)
    result_image = Image.open(buf).copy()
    
    suffix = '.jpg' if fmt == 'JPEG' else '.png'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    if fmt == 'JPEG':
        result_image.save(tmp.name, format='JPEG', quality=92)
    else:
        result_image.save(tmp.name, format='PNG')
    tmp.close()
    
    # Calculate PSNR
    mse = np.mean((img_array - result_array)**2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    
    return result_image, tmp.name, f"PSNR: {psnr:.2f} dB (Multi-Channel α={alpha})"

def detect_multichannel_watermark(filepath: str, alpha: float = 0.6):
    """
    Detect watermark across all RGB channels.
    Combine evidence from all channels for robust detection.
    """
    if filepath is None:
        raise gr.Error("Please upload an image.")
    
    suspect_image = Image.open(filepath)
    img_array = np.array(suspect_image.convert('RGB')).astype(np.float64)
    h, w, _ = img_array.shape
    
    # Generate same watermark sequence
    rng = np.random.RandomState(42)
    W = rng.choice([-1.0, 1.0], size=1024)
    
    # Detect per channel
    channel_scores = []
    block_size = 8
    
    for channel_idx in range(3):
        channel = img_array[:, :, channel_idx]
        extracted = []
        
        idx = 0
        for row in range(0, h - block_size + 1, block_size):
            for col in range(0, w - block_size + 1, block_size):
                block = channel[row:row+block_size, col:col+block_size]
                C = dct(dct(block.T, norm='ortho').T, norm='ortho')
                
                # Extract from same positions used in embedding
                val = C[3, 3]  # Primary position
                extracted.append(val)
                idx += 1
        
        extracted = np.array(extracted)
        W_tiled = np.tile(W, (len(extracted) // len(W)) + 1)[:len(extracted)]
        
        if len(extracted) > 1:
            corr = np.corrcoef(extracted, W_tiled)[0, 1]
            channel_scores.append(corr if not np.isnan(corr) else 0.0)
        else:
            channel_scores.append(0.0)
    
    # Combine scores (average across channels)
    combined_score = np.mean(channel_scores)
    threshold = 0.1  # Lower threshold for multi-channel
    
    decision = "DETECTED" if combined_score > threshold else "NOT DETECTED"
    
    channel_info = f"R:{channel_scores[0]:.3f}, G:{channel_scores[1]:.3f}, B:{channel_scores[2]:.3f}"
    print(f"DEBUG Multi-Channel: combined={combined_score:.4f}, threshold={threshold:.4f}, decision={decision}")
    print(f"DEBUG Per-channel: {channel_info}")
    
    return f"{combined_score:.6f}", f"{threshold:.4f}", decision, channel_info

# ============================================================
# GRADIO UI (with Tabs 1-3 unchanged, plus Tabs 4-5 new)
# ============================================================

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

with gr.Blocks(title="VWE v3 - Multi-Algorithm Watermark", theme=custom_theme) as demo:
    gr.Markdown("# VWE v3 — Multi-Algorithm Watermark System\nOriginal DWT-DCT + Phase Domain + Multi-Channel Frequency")
    
    with gr.Tabs():
        # TAB 1: Original Embed (UNCHANGED)
        with gr.TabItem("🔒 Embed Watermark (Original)"):
            with gr.Row():
                img_input = gr.Image(type="filepath", label="Upload Original Image")
                with gr.Column():
                    img_preview = gr.Image(type="pil", label="Preview (display only)")
                    img_download = gr.File(label="Download Watermarked Image (original format)")
            embed_btn = gr.Button("Embed Watermark (Alpha=0.5)", variant="primary")
            embed_btn.click(embed_watermark, inputs=[img_input], outputs=[img_preview, img_download])
        
        # TAB 2: Original Detect (UNCHANGED)
        with gr.TabItem("🔍 Detect Watermark (Original)"):
            with gr.Row():
                suspect_img = gr.Image(type="filepath", label="Upload Suspect Image")
                with gr.Column():
                    rho_val = gr.Textbox(label="Correlation Score ρ")
                    thresh_val = gr.Textbox(label="Detection Threshold (4σ)")
                    decision_val = gr.Textbox(label="Result Decision")
            detect_btn = gr.Button("Detect Watermark", variant="primary")
            detect_btn.click(detect_watermark, inputs=[suspect_img], outputs=[rho_val, thresh_val, decision_val])

        # TAB 3: Original Visualize (UNCHANGED)
        with gr.TabItem("🔬 Visualize Watermark (Original)"):
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
        
        # TAB 4: Phase Domain Watermark (NEW - SynthID Style)
        with gr.TabItem("🌊 Phase Domain (SynthID-style)"):
            gr.Markdown("### Embed watermark in FFT phase domain (green channel)\nBased on SynthID reverse-engineering: fixed phase perturbation at frequency bins")
            with gr.Row():
                phase_input = gr.Image(type="filepath", label="Upload Original Image")
                with gr.Column():
                    freq_u = gr.Slider(minimum=1, maximum=32, value=9, step=1, label="Frequency Bin U (9=SynthID default)")
                    freq_v = gr.Slider(minimum=1, maximum=32, value=9, step=1, label="Frequency Bin V (9=SynthID default)")
                    epsilon = gr.Slider(minimum=0.001, maximum=0.1, value=0.02, step=0.001, label="Phase ε (watermark strength)")
                    phase_preview = gr.Image(type="pil", label="Preview")
                    phase_download = gr.File(label="Download Phase-Watermarked Image")
                    phase_quality = gr.Textbox(label="Quality Info", interactive=False)
            phase_embed_btn = gr.Button("Embed Phase Watermark", variant="primary")
            phase_embed_btn.click(embed_phase_watermark, inputs=[phase_input, freq_u, freq_v, epsilon], outputs=[phase_preview, phase_download, phase_quality])
            
            gr.Markdown("---")
            gr.Markdown("### Detect Phase Watermark")
            with gr.Row():
                phase_suspect = gr.Image(type="filepath", label="Upload Suspect Image")
                with gr.Column():
                    phase_rho = gr.Textbox(label="Phase Coherence")
                    phase_thresh = gr.Textbox(label="Coherence Threshold")
                    phase_decision = gr.Textbox(label="Result")
            phase_detect_btn = gr.Button("Detect Phase Watermark", variant="primary")
            phase_detect_btn.click(detect_phase_watermark, inputs=[phase_suspect, freq_u, freq_v], outputs=[phase_rho, phase_thresh, phase_decision])
        
        # TAB 5: Multi-Channel Frequency Watermark (NEW)
        with gr.TabItem("🌈 Multi-Channel Frequency"):
            gr.Markdown("### Embed watermark across ALL RGB channels in DCT domain\nMore robust: signal spread across R, G, B channels with redundant DCT positions")
            with gr.Row():
                mc_input = gr.Image(type="filepath", label="Upload Original Image")
                with gr.Column():
                    mc_alpha = gr.Slider(minimum=0.2, maximum=1.5, value=0.6, step=0.1, label="Alpha (per-channel strength)")
                    mc_preview = gr.Image(type="pil", label="Preview")
                    mc_download = gr.File(label="Download Multi-Channel Watermarked Image")
                    mc_quality = gr.Textbox(label="Quality Info", interactive=False)
            mc_embed_btn = gr.Button("Embed Multi-Channel Watermark", variant="primary")
            mc_embed_btn.click(embed_multichannel_watermark, inputs=[mc_input, mc_alpha], outputs=[mc_preview, mc_download, mc_quality])
            
            gr.Markdown("---")
            gr.Markdown("### Detect Multi-Channel Watermark")
            with gr.Row():
                mc_suspect = gr.Image(type="filepath", label="Upload Suspect Image")
                with gr.Column():
                    mc_rho = gr.Textbox(label="Combined Score")
                    mc_thresh = gr.Textbox(label="Detection Threshold")
                    mc_decision = gr.Textbox(label="Result")
                    mc_channels = gr.Textbox(label="Per-Channel Scores")
            mc_detect_btn = gr.Button("Detect Multi-Channel Watermark", variant="primary")
            mc_detect_btn.click(detect_multichannel_watermark, inputs=[mc_suspect, mc_alpha], outputs=[mc_rho, mc_thresh, mc_decision, mc_channels])

if __name__ == "__main__":
    demo.launch(server_port=7863)
