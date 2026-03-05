import os
import glob
import numpy as np
import pywt
import cv2
from scipy.fftpack import dct, idct
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

# --- IMPROVED CORE LOGIC (v2 - Multi-Band for LoRA Survival) ---

def generate_fixed_pn_sequence(length: int, seed: int = 42) -> np.ndarray:
    """Generate the fixed pseudo-noise sequence W of +1/-1 values using numpy seed."""
    rng = np.random.RandomState(seed)
    return rng.choice([-1.0, 1.0], size=length)

def apply_2level_dwt_full(Y_channel: np.ndarray):
    """Decompose the luminance channel into 2-level Haar DWT - return ALL subbands."""
    coeffs1 = pywt.dwt2(Y_channel, 'haar')
    LL1, (LH1, HL1, HH1) = coeffs1
    coeffs2 = pywt.dwt2(LL1, 'haar')
    LL2, (LH2, HL2, HH2) = coeffs2
    return LL2, LH2, HL2, HH2, coeffs1, coeffs2

def reconstruct_from_dwt_full(LL2_mod, LH2_mod, HL2_mod, HH2_mod, coeffs1, coeffs2):
    """Reconstruct modified Y channel from ALL updated 2nd-level subbands."""
    LL1_orig, (LH1, HL1, HH1) = coeffs1
    LL2_orig, (LH2_orig, HL2_orig, HH2_orig) = coeffs2
    
    LL1_new = pywt.idwt2((LL2_mod, (LH2_mod, HL2_mod, HH2_mod)), 'haar')
    Y_new = pywt.idwt2((LL1_new, (LH1, HL1, HH1)), 'haar')
    return Y_new

def embed_block_multi(block: np.ndarray, w_bit: float, alpha: float = 0.8):
    """Embed watermark bit into multiple DCT positions for redundancy."""
    dct_positions = [(3, 3), (2, 2), (4, 4)]
    C = dct(dct(block.T, norm='ortho').T, norm='ortho')
    block_modified = block.copy()
    
    for pos in dct_positions:
        i, j = pos
        if i < block.shape[0] and j < block.shape[1]:
            effective_coeff = max(abs(C[i, j]), 8.0)
            C[i, j] = C[i, j] + alpha * w_bit * effective_coeff
    
    block_modified = idct(idct(C.T, norm='ortho').T, norm='ortho')
    return block_modified

def embed_single_image_v2(src_path: str, dst_path: str, alpha: float = 0.8, seed: int = 42) -> bool:
    """Embed watermark across ALL 2nd-level subbands for LoRA survival."""
    try:
        pil_image = Image.open(src_path)
        
        if pil_image.width < 64 or pil_image.height < 64:
            print(f"Skipping {os.path.basename(src_path)}: Image is too small (<64x64).")
            return False
        
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
            W = generate_fixed_pn_sequence(4096, seed)
            if len(W) < N_blocks:
                W = np.tile(W, (N_blocks // len(W)) + 1)[:N_blocks]
            
            # Embed in this subband
            subband_wm = subband.copy()
            idx = 0
            for row in range(0, h - block_size + 1, block_size):
                for col in range(0, w - block_size + 1, block_size):
                    block = subband_wm[row:row+block_size, col:col+block_size]
                    block_modified = embed_block_multi(block, W[idx % len(W)], alpha)
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
        
        # Save as PNG (lossless to preserve watermark)
        result_image.save(dst_path, format='PNG')
        
        # Calculate PSNR for quality check
        orig_array = np.array(pil_image.convert('RGB')).astype(float)
        wm_array = np.array(result_image).astype(float)
        mse = np.mean((orig_array - wm_array)**2)
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
        
        print(f"Watermarked {os.path.basename(src_path)} -> PSNR: {psnr:.2f} dB")
        return True
        
    except Exception as e:
        print(f"Exception while watermarking {os.path.basename(src_path)}: {e}")
        return False

def main():
    src_dir = "dataset pony ip"
    dst_dir = "watermarked ones v2"
    alpha = 0.8  # Stronger embedding for LoRA survival
    
    os.makedirs(dst_dir, exist_ok=True)
    
    # Gather matching image extensions
    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    src_files = []
    for ext in exts:
        src_files.extend(glob.glob(os.path.join(src_dir, ext)))
    
    src_files = sorted(list(set(src_files)))
    
    if not src_files:
        print(f"No image files found in '{src_dir}'.")
        return
    
    tasks = []
    for src in src_files:
        base_name = os.path.basename(src)
        name, _ = os.path.splitext(base_name)
        dst = os.path.join(dst_dir, f"{name}.png")
        tasks.append((src, dst))
    
    print(f"-> Launching parallel processor (v2 multi-band) for {len(tasks)} images...")
    print(f"-> Alpha = {alpha} (stronger for LoRA survival)")
    
    success_count = 0
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(embed_single_image_v2, src, dst, alpha): src for src, dst in tasks}
        for future in futures:
            src_file = futures[future]
            try:
                res = future.result()
                if res:
                    success_count += 1
            except Exception as exc:
                print(f"Task generated an exception for {os.path.basename(src_file)}: {exc}")
    
    print(f"-> Batch processing finished!")
    print(f"-> Successfully watermarked {success_count} / {len(tasks)} files.")
    print(f"-> Outputs saved in: '{dst_dir}' directory.")

if __name__ == "__main__":
    main()
