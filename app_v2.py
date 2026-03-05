import os
import tempfile
import gradio as gr
import numpy as np
import pywt
from scipy.fftpack import dct, idct
from PIL import Image
import cv2
from io import BytesIO

# --- IMPROVED CORE UTILITIES FOR LoRA SURVIVAL ---

def generate_fixed_pn_sequence(length: int, seed: int = 42) -> np.ndarray:
    """Generate a fixed pseudo-noise sequence W of +1/-1 values using numpy seed."""
    rng = np.random.RandomState(seed)
    return rng.choice([-1.0, 1.0], size=length)

def apply_2level_dwt_full(Y_channel: np.ndarray):
    """Decompose the luminance channel into 2-level Haar DWT - return ALL subbands."""
    coeffs1 = pywt.dwt2(Y_channel, 'haar')
    LL1, (LH1, HL1, HH1) = coeffs1
    coeffs2 = pywt.dwt2(LL1, 'haar')
    LL2, (LH2, HL2, HH2) = coeffs2
    # Return all subbands for multi-band embedding
    return LL2, LH2, HL2, HH2, coeffs1, coeffs2

def reconstruct_from_dwt_full(LL2_mod, LH2_mod, HL2_mod, HH2_mod, coeffs1, coeffs2):
    """Reconstruct modified Y channel from ALL updated 2nd-level subbands."""
    LL1_orig, (LH1, HL1, HH1) = coeffs1
    LL2_orig, (LH2_orig, HL2_orig, HH2_orig) = coeffs2
    
    # Reconstruct LL1 from modified 2nd-level subbands
    LL1_new = pywt.idwt2((LL2_mod, (LH2_mod, HL2_mod, HH2_mod)), 'haar')
    
    # Reconstruct Y from LL1 and original 1st-level detail subbands
    Y_new = pywt.idwt2((LL1_new, (LH1, HL1, HH1)), 'haar')
    return Y_new

def embed_block(block: np.ndarray, w_bit: float, alpha: float = 0.8, dct_positions: list = None):
    """Embed watermark bit into multiple DCT positions for redundancy."""
    if dct_positions is None:
        dct_positions = [(3, 3), (2, 2), (4, 4)]  # Multiple positions for robustness
    
    C = dct(dct(block.T, norm='ortho').T, norm='ortho')
    block_modified = block.copy()
    
    for pos in dct_positions:
        i, j = pos
        if i < block.shape[0] and j < block.shape[1]:
            effective_coeff = max(abs(C[i, j]), 8.0)
            C[i, j] = C[i, j] + alpha * w_bit * effective_coeff
    
    block_modified = idct(idct(C.T, norm='ortho').T, norm='ortho')
    return block_modified

def extract_block(block: np.ndarray, dct_positions: list = None):
    """Extract watermark signs from multiple DCT positions."""
    if dct_positions is None:
        dct_positions = [(3, 3), (2, 2), (4, 4)]
    
    C = dct(dct(block.T, norm='ortho').T, norm='ortho')
    signs = []
    for pos in dct_positions:
        i, j = pos
        if i < block.shape[0] and j < block.shape[1]:
            signs.append(C[i, j])
    return signs

# --- TAB 1: EMBED (Improved for LoRA Survival) ---

def embed_watermark_v2(filepath: str, alpha: float = 0.8, seed: int = 42):
    """Embed watermark across ALL 2nd-level wavelet subbands for LoRA survival."""
    if filepath is None:
        raise gr.Error("Please upload an image.")
    
    # Detect original format
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
    
    # 2-level Haar DWT - get ALL subbands
    LL2, LH2, HL2, HH2, coeffs1, coeffs2 = apply_2level_dwt_full(Y)
    
    # Process each subband
    subbands = [('LL2', LL2), ('LH2', LH2), ('HL2', HL2), ('HH2', HH2)]
    modified_subbands = {}
    block_size = 8
    
    for name, subband in subbands:
        h, w = subband.shape
        N_blocks = (h // block_size) * (w // block_size)
        
        # Generate/tile W for this subband
        W = generate_fixed_pn_sequence(4096, seed)  # Longer sequence for more entropy
        if len(W) < N_blocks:
            W = np.tile(W, (N_blocks // len(W)) + 1)[:N_blocks]
        
        # Embed in this subband
        subband_wm = subband.copy()
        idx = 0
        for row in range(0, h - block_size + 1, block_size):
            for col in range(0, w - block_size + 1, block_size):
                block = subband_wm[row:row+block_size, col:col+block_size]
                block_modified = embed_block(block, W[idx % len(W)], alpha)
                subband_wm[row:row+block_size, col:col+block_size] = block_modified
                idx += 1
        
        modified_subbands[name] = subband_wm
    
    # Reconstruct Y from ALL modified subbands
    Y_wm = reconstruct_from_dwt_full(
        modified_subbands['LL2'],
        modified_subbands['LH2'],
        modified_subbands['HL2'],
        modified_subbands['HH2'],
        coeffs1, coeffs2
    )
    Y_wm = np.clip(Y_wm, 0, 255)
    
    # Merge back
    img_wm = np.stack([Y_wm, Cr, Cb], axis=2).astype(np.uint8)
    img_wm_rgb = cv2.cvtColor(img_wm, cv2.COLOR_YCrCb2RGB)
    result_image = Image.fromarray(img_wm_rgb)
    
    # Save output
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
    
    # Calculate PSNR for quality check
    mse = np.mean((np.array(pil_image.convert('RGB')).astype(float) - np.array(result_image).astype(float))**2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    
    return result_image, tmp.name, f"PSNR: {psnr:.2f} dB (alpha={alpha})"

# --- TAB 2: DETECT (Improved with Multi-Band) ---

def detect_watermark_v2(filepath: str, seed: int = 42):
    """Detect watermark across ALL 2nd-level subbands."""
    if filepath is None:
        raise gr.Error("Please upload an image to analyze.")
    
    suspect_image = Image.open(filepath)
    if suspect_image.width < 64 or suspect_image.height < 64:
        raise gr.Error("Image must be at least 64x64 pixels.")
    
    img = suspect_image.convert('RGB')
    img_array = np.array(img, dtype=np.float32)
    img_cv = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float64)
    Y = img_cv[:,:,0]
    
    # Apply DWT - get ALL subbands
    LL2, LH2, HL2, HH2, _, _ = apply_2level_dwt_full(Y)
    
    # Extract from all subbands
    subbands = [LL2, LH2, HL2, HH2]
    subband_names = ['LL2', 'LH2', 'HL2', 'HH2']
    block_size = 8
    
    all_signs = []
    for subband, name in zip(subbands, subband_names):
        h, w = subband.shape
        N_blocks = (h // block_size) * (w // block_size)
        
        for row in range(0, h - block_size + 1, block_size):
            for col in range(0, w - block_size + 1, block_size):
                block = subband[row:row+block_size, col:col+block_size]
                signs = extract_block(block)
                all_signs.extend(signs)
    
    extracted_signs = np.array(all_signs)
    N_actual = len(extracted_signs)
    
    if N_actual == 0:
        return "0.0", "0.0", "ERROR: No blocks analyzed", "0"
    
    # Regenerate W (use same longer sequence)
    W = generate_fixed_pn_sequence(4096, seed)
    if len(W) < N_actual:
        W = np.tile(W, (N_actual // len(W)) + 1)[:N_actual]
    
    # Compute correlation per subband for detailed analysis
    correlations = []
    idx = 0
    for subband, name in zip(subbands, subband_names):
        h, w = subband.shape
        N_blocks = (h // block_size) * (w // block_size)
        end_idx = idx + N_blocks * 3  # 3 DCT positions per block
        
        sub_signs = extracted_signs[idx:end_idx]
        sub_W = W[idx:end_idx]
        
        if len(sub_signs) > 1 and len(sub_W) > 1:
            corr_matrix = np.corrcoef(sub_signs[:len(sub_W)], sub_W[:len(sub_signs)])
            rho = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
            correlations.append(f"{name}: {rho:.4f}")
        
        idx = end_idx
    
    # Overall correlation
    corr_matrix = np.corrcoef(extracted_signs[:N_actual], W[:N_actual])
    rho = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
    threshold = 4.0 / np.sqrt(N_actual)
    
    decision = "DETECTED" if rho > threshold else "NOT DETECTED"
    
    subband_info = " | ".join(correlations)
    print(f"DEBUG: rho={rho:.6f}, threshold={threshold:.6f}, N={N_actual}, decision={decision}")
    print(f"DEBUG Subbands: {subband_info}")
    
    return f"{rho:.6f}", f"{threshold:.6f}", decision, f"{N_actual} blocks"

# --- GRADIO UI (Updated) ---

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

with gr.Blocks(title="VWE v2 - LoRA-Survivable Watermark", theme=custom_theme) as demo:
    gr.Markdown("# VWE v2 — LoRA-Survivable Watermark\nMulti-band DWT-DCT embedding for training data leakage detection.")
    
    with gr.Tabs():
        with gr.TabItem("🔒 Embed Watermark (v2)"):
            with gr.Row():
                img_input = gr.Image(type="filepath", label="Upload Original Image")
                with gr.Column():
                    alpha_slider = gr.Slider(minimum=0.5, maximum=1.5, value=0.8, step=0.1, label="Alpha (Embedding Strength)")
                    img_preview = gr.Image(type="pil", label="Preview")
                    img_download = gr.File(label="Download Watermarked Image")
                    quality_info = gr.Textbox(label="Quality Info", interactive=False)
            embed_btn = gr.Button("Embed Watermark (Multi-Band)", variant="primary")
            embed_btn.click(embed_watermark_v2, inputs=[img_input, alpha_slider], outputs=[img_preview, img_download, quality_info])
        
        with gr.TabItem("🔍 Detect Watermark (v2)"):
            with gr.Row():
                suspect_img = gr.Image(type="filepath", label="Upload Suspect Image")
                with gr.Column():
                    rho_val = gr.Textbox(label="Correlation Score ρ")
                    thresh_val = gr.Textbox(label="Detection Threshold")
                    decision_val = gr.Textbox(label="Result")
                    blocks_val = gr.Textbox(label="Blocks Analyzed")
            detect_btn = gr.Button("Detect Watermark (Multi-Band)", variant="primary")
            detect_btn.click(detect_watermark_v2, inputs=[suspect_img], outputs=[rho_val, thresh_val, decision_val, blocks_val])

if __name__ == "__main__":
    demo.launch(server_port=7862)
