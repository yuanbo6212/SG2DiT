# File: evaluate_metrics.py
import os
import re
import csv
import cv2
import math
import numpy as np
from glob import glob
from tqdm import tqdm
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mutual_info_score
from scipy.ndimage import gaussian_filter

# ---------------------------
# Utilities
# ---------------------------
ALLOWED_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


def list_images(folder):
    imgs = []
    for ext in ALLOWED_EXTS:
        imgs.extend(glob(os.path.join(folder, ext)))
    return sorted(imgs)


def image_read_cv2(path, mode='RGB'):
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    im = im.astype(np.float32)
    if mode == 'RGB':
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("mode must be 'RGB' or 'GRAY'")


def resize_to_target(src, target_shape):
    th, tw = target_shape[:2]
    sh, sw = src.shape[:2]
    if (sh, sw) == (th, tw):
        return src
    interp = cv2.INTER_AREA if sh > th or sw > tw else cv2.INTER_LINEAR
    resized = cv2.resize(src, (tw, th), interpolation=interp)
    return resized.astype(np.float32)


def normalize_name_for_match(path_or_name):
    name = os.path.basename(path_or_name)
    name = os.path.splitext(name)[0].lower()
    for tok in ["flir_", "flir-", "flir", "_ir", "-ir", "ir_", "_vi", "-vi", "vi_", "vis_", "rgb_", "visible_"]:
        name = name.replace(tok, "")
    name = re.sub(r'[^a-z0-9]', '', name)
    return name


def extract_digits(s):
    m = re.search(r'(\d+)', s)
    return m.group(1) if m else None


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------
# Evaluator class
# ---------------------------
class Evaluator:
    @classmethod
    def _to_gray_uint8(cls, img):
        if img.ndim == 3:
            img8 = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2GRAY)
        else:
            img8 = np.uint8(np.round(img))
        return img8


    @classmethod
    def EN(cls, img):
        gray = cls._to_gray_uint8(img)
        hist = np.bincount(gray.flatten(), minlength=256).astype(np.float64)
        prob = hist / hist.sum()
        prob = prob[prob > 0]
        return float(-np.sum(prob * np.log2(prob)))

    @classmethod
    def SD(cls, img):
        return float(np.std(img))

    @classmethod
    def SF(cls, img):
        gray = cls._to_gray_uint8(img)
        fx = np.mean((gray[:, 1:] - gray[:, :-1]) ** 2)
        fy = np.mean((gray[1:, :] - gray[:-1, :]) ** 2)
        return float(np.sqrt(fx + fy))

    @classmethod
    def AG(cls, img):
        gray = cls._to_gray_uint8(img).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx ** 2 + gy ** 2)
        return float(np.mean(grad))


    @classmethod
    def MI(cls, F, A, B):
        Fg = cls._to_gray_uint8(F).flatten()
        Ag = cls._to_gray_uint8(A).flatten()
        Bg = cls._to_gray_uint8(B).flatten()
        return float(mutual_info_score(Fg, Ag) + mutual_info_score(Fg, Bg))

    @classmethod
    def MSE(cls, F, A, B):
        m1 = np.mean((A.astype(np.float32) - F.astype(np.float32)) ** 2)
        m2 = np.mean((B.astype(np.float32) - F.astype(np.float32)) ** 2)
        return float((m1 + m2) / 2.0)

    @classmethod
    def CC(cls, F, A, B):
        f = F.flatten().astype(np.float64)
        a = A.flatten().astype(np.float64)
        b = B.flatten().astype(np.float64)
        rAF = np.corrcoef(a, f)[0, 1] if a.std() > 0 and f.std() > 0 else 0.0
        rBF = np.corrcoef(b, f)[0, 1] if b.std() > 0 and f.std() > 0 else 0.0
        return float((rAF + rBF) / 2.0)

    @classmethod
    def PSNR(cls, F, A, B):
        mse = cls.MSE(F, A, B)
        if mse == 0:
            return float('inf')
        return float(10 * np.log10((255.0 ** 2) / mse))

    @classmethod
    def SSIM(cls, F, A, B):
        if F.ndim == 2:
            s1 = ssim(F.astype('uint8'), A.astype('uint8'), data_range=255)
            s2 = ssim(F.astype('uint8'), B.astype('uint8'), data_range=255)
        else:
            s1 = ssim(F.astype('uint8'), A.astype('uint8'), data_range=255, channel_axis=-1)
            s2 = ssim(F.astype('uint8'), B.astype('uint8'), data_range=255, channel_axis=-1)
        return float(s1 + s2)

    @classmethod
    def Qabf(cls, F, A, B):
        Fg = cls._to_gray_uint8(F)
        Ag = cls._to_gray_uint8(A)
        Bg = cls._to_gray_uint8(B)
        gxA, gyA = np.gradient(Ag.astype(np.float32))
        gxB, gyB = np.gradient(Bg.astype(np.float32))
        gxF, gyF = np.gradient(Fg.astype(np.float32))
        magA = np.sqrt(gxA ** 2 + gyA ** 2)
        magB = np.sqrt(gxB ** 2 + gyB ** 2)
        magF = np.sqrt(gxF ** 2 + gyF ** 2)
        pa = magA / (magA + magB + 1e-8)
        pb = magB / (magA + magB + 1e-8)
        Q = pa * (magF / (magA + 1e-8)) + pb * (magF / (magB + 1e-8))
        return float(np.mean(Q))

    @classmethod
    def VIF(cls, F, A):
        A = cls._to_gray_uint8(A).astype(np.float32)
        F = cls._to_gray_uint8(F).astype(np.float32)
        sigma_nsq = 2.0
        num = 0.0
        den = 0.0
        for scale in range(4):
            N = 2 ** (4 - scale)
            A1 = gaussian_filter(A, N)
            F1 = gaussian_filter(F, N)
            sigma_A = np.std(A1)
            sigma_F = np.std(F1)
            sigma_AF = np.mean((A1 - np.mean(A1)) * (F1 - np.mean(F1)))
            g = sigma_AF / (sigma_A ** 2 + 1e-8)
            sv_sq = sigma_F ** 2 - g * sigma_AF
            num += np.log10(1 + (g ** 2 * sigma_A ** 2) / (sv_sq + sigma_nsq))
            den += np.log10(1 + sigma_A ** 2 / sigma_nsq)
        return float(num / (den + 1e-8))

    @classmethod
    def SCD(cls, F, A, B):
        Fg = cls._to_gray_uint8(F)
        Ag = cls._to_gray_uint8(A)
        Bg = cls._to_gray_uint8(B)
        gradF = np.gradient(Fg.astype(np.float32))
        gradA = np.gradient(Ag.astype(np.float32))
        gradB = np.gradient(Bg.astype(np.float32))
        scdA = np.mean(np.abs(gradF[0] - gradA[0]) + np.abs(gradF[1] - gradA[1]))
        scdB = np.mean(np.abs(gradF[0] - gradB[0]) + np.abs(gradF[1] - gradB[1]))
        return float(1 - (scdA + scdB) / (2 * 255))


# ---------------------------
# Fusion-folder evaluation
# ---------------------------
def evaluate_fusion_folder(imgA_dir, imgB_dir, imgF_dir, mode='RGB', csv_out=None, try_numeric_match=True):
    evaluator = Evaluator()
    imgF_list = list_images(imgF_dir)
    if not imgF_list:
        raise ValueError(f"No fusion images found in {imgF_dir}")

    def build_lookup(folder):
        lookup = {}
        files = list_images(folder)
        for p in files:
            key = normalize_name_for_match(p)
            lookup.setdefault(key, p)
        return lookup

    ir_dict = build_lookup(imgA_dir)
    vi_dict = build_lookup(imgB_dir)
    ir_digits = {extract_digits(normalize_name_for_match(p)): p for p in list_images(imgA_dir) if extract_digits(normalize_name_for_match(p))}
    vi_digits = {extract_digits(normalize_name_for_match(p)): p for p in list_images(imgB_dir) if extract_digits(normalize_name_for_match(p))}

    per_results = []
    matched = 0
    resized_count = 0

    print(f"\n[融合评估] 共 {len(imgF_list)} 组候选融合图像")
    for imgF_path in tqdm(imgF_list, desc="Matching & computing"):
        key = normalize_name_for_match(imgF_path)
        imgA_path = ir_dict.get(key)
        imgB_path = vi_dict.get(key)
        if (imgA_path is None or imgB_path is None) and try_numeric_match:
            d = extract_digits(key)
            if d:
                imgA_path = imgA_path or ir_digits.get(d)
                imgB_path = imgB_path or vi_digits.get(d)
        if imgA_path is None or imgB_path is None:
            continue

        imgA = image_read_cv2(imgA_path, mode=mode)
        imgB = image_read_cv2(imgB_path, mode=mode)
        imgF = image_read_cv2(imgF_path, mode=mode)
        target_shape = imgA.shape[:2]
        if imgB.shape[:2] != target_shape:
            imgB = resize_to_target(imgB, target_shape)
            resized_count += 1
        if imgF.shape[:2] != target_shape:
            imgF = resize_to_target(imgF, target_shape)
            resized_count += 1


        en = evaluator.EN(imgF)
        sd = evaluator.SD(imgF)
        sf = evaluator.SF(imgF)
        ag = evaluator.AG(imgF)


        mi = evaluator.MI(imgF, imgA, imgB)
        mse = evaluator.MSE(imgF, imgA, imgB)
        cc = evaluator.CC(imgF, imgA, imgB)
        psnr = evaluator.PSNR(imgF, imgA, imgB)
        ssim_val = evaluator.SSIM(imgF, imgA, imgB)
        qabf = evaluator.Qabf(imgF, imgA, imgB)
        vif = evaluator.VIF(imgF, imgA)
        scd = evaluator.SCD(imgF, imgA, imgB)

        per_results.append({
            'name': os.path.basename(imgF_path),
            'EN': en, 'SD': sd, 'SF': sf, 'AG': ag,
            'MI': mi, 'MSE': mse, 'CC': cc, 'PSNR': psnr,
            'SSIM': ssim_val, 'Qabf': qabf, 'VIF': vif, 'SCD': scd
        })
        matched += 1

    if matched == 0:
        raise ValueError("未匹配到任何图像对，请检查文件名或路径。")

    avg = {k: float(np.nanmean([r[k] for r in per_results])) for k in per_results[0] if k != 'name'}
    print(f"\n成功匹配 {matched} 组图像（调整尺寸 {resized_count} 次），平均结果：")
    for k, v in avg.items():
        print(f"{k}: {v:.4f}")

    if csv_out:
        with open(csv_out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(per_results[0].keys()))
            writer.writeheader()
            writer.writerows(per_results)
        print(f"结果已保存到 {csv_out}")

    return per_results, avg


# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    imgA_folder = "ir"
    imgB_folder = "vis"
    imgF_folder = ""
    mode = 'RGB'
    csv_out = os.path.join(imgF_folder, f"fusion_metrics_{timestamp()}.csv")

    _, avg = evaluate_fusion_folder(imgA_folder, imgB_folder, imgF_folder, mode=mode, csv_out=csv_out)
    print("\nDone.")
