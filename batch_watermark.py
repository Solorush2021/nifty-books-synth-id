import os
import glob
import numpy as np
import pywt
import cv2
from scipy.fftpack import dct, idct
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

# --- REPLICATED ENGINE CORE LOGIC ---

def generate_fixed_pn_sequence(length: int) -> np.ndarray:
    """Generate the fixed pseudo-noise sequence W of +1/-1 values using numpy seed 42."""
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

def embed_single_image(src_path: str, dst_path: str) -> bool:
    """Embed the watermark in a single image and save as lossless PNG."""
    try:
        pil_image = Image.open(src_path)
        
        # Enforce minimum size restriction
        if pil_image.width < 64 or pil_image.height < 64:
            print(f"Skipping {os.path.basename(src_path)}: Image is too small (<64x64).")
            return False
            
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
                
                # Embedding formula with alpha=0.5 and minimum floor
                effective_coeff = max(abs(C[3, 3]), 8.0)
                C[3, 3] = C[3, 3] + 0.5 * W[idx % len(W)] * effective_coeff
                
                block_modified = idct(idct(C.T, norm='ortho').T, norm='ortho')
                LL2_wm[row:row+block_size, col:col+block_size] = block_modified
                idx += 1
                
        # Reconstruct Y
        Y_wm = reconstruct_from_dwt(LL2_wm, coeffs1, coeffs2)
        Y_wm = np.clip(Y_wm, 0, 255)
        
        # Merge back (retaining watermarked luminance)
        img_wm = np.stack([Y_wm, Cr, Cb], axis=2).astype(np.uint8)
        img_wm_rgb = cv2.cvtColor(img_wm, cv2.COLOR_YCrCb2RGB)
        result_image = Image.fromarray(img_wm_rgb)
        
        # Save as PNG format
        result_image.save(dst_path, format='PNG')
        return True
    except Exception as e:
        print(f"Exception while watermarking {os.path.basename(src_path)}: {e}")
        return False

def main():
    src_dir = "dataset pony ip"
    dst_dir = "watermarked ones"
    os.makedirs(dst_dir, exist_ok=True)
    
    # Gather matching image extensions
    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    src_files = []
    for ext in exts:
        src_files.extend(glob.glob(os.path.join(src_dir, ext)))
    
    src_files = sorted(list(set(src_files))) # unique entries
    
    if not src_files:
        print(f"No image files found in '{src_dir}'.")
        return
        
    tasks = []
    for src in src_files:
        base_name = os.path.basename(src)
        name, _ = os.path.splitext(base_name)
        # Always save as .png to prevent JPEG compression artifacts
        dst = os.path.join(dst_dir, f"{name}.png")
        tasks.append((src, dst))
        
    print(f"-> Launching parallel processor for {len(tasks)} images...")
    
    success_count = 0
    with ProcessPoolExecutor() as executor:
        # Submit tasks in parallel
        futures = {executor.submit(embed_single_image, src, dst): src for src, dst in tasks}
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
    print(f"-> Outputs saved cleanly in: '{dst_dir}' directory.")

if __name__ == "__main__":
    main()
