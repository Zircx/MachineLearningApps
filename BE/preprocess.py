import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from skimage.measure import regionprops, label as sk_label


def extract_features(img_bytes: bytes, img_size: int = 64):
    nparr   = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Gambar tidak dapat dibaca. Pastikan format PNG/JPG.")

    img_bgr  = cv2.resize(img_bgr, (img_size, img_size))
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    features = []

    #  1. Color Mean & Std (6 features)
    color_vals = []
    for ch_idx in range(3):
        ch = img_rgb[:, :, ch_idx].astype(np.float32)
        mean_val = ch.mean()
        std_val  = ch.std()
        features += [mean_val, std_val]
        color_vals.append({"mean": float(mean_val), "std": float(std_val)})

    # 2. Color Histogram 8-bin per channel (24 features)
    hist_vals = []
    for ch_idx in range(3):
        hist, _ = np.histogram(img_rgb[:, :, ch_idx], bins=8, range=(0, 256))
        hist     = hist.astype(np.float32) / hist.sum()
        features += hist.tolist()
        hist_vals.append(hist.tolist())

    #  3. GLCM Texture (5 fitur) 
    glcm = graycomatrix(img_gray, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    glcm_vals = {}
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        val = float(graycoprops(glcm, prop)[0, 0])
        features.append(val)
        glcm_vals[prop] = val

    # 4. LBP (10 features) 
    lbp         = local_binary_pattern(img_gray, P=8, R=1.0, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    lbp_hist    = lbp_hist.astype(np.float32) / lbp_hist.sum()
    features   += lbp_hist.tolist()
    lbp_uniformity = float(lbp_hist.max())
    lbp_entropy    = float(-np.sum(lbp_hist * np.log2(lbp_hist + 1e-10)))

    # 5. HOG (36 features) 
    hog_feats = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=False, feature_vector=True)
    hog_sub   = hog_feats[:36] if len(hog_feats) >= 36 else \
                np.pad(hog_feats, (0, 36 - len(hog_feats)))
    features += hog_sub.tolist()
    hog_mean  = float(hog_sub.mean())
    hog_max   = float(hog_sub.max())

    #  6. Morphological (3 features) 
    _, thresh = cv2.threshold(img_gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    labeled = sk_label(thresh)
    props   = regionprops(labeled)
    if props:
        largest      = max(props, key=lambda r: r.area)
        area         = largest.area      / (img_size * img_size)
        perimeter    = largest.perimeter / (4 * img_size)
        eccentricity = largest.eccentricity
    else:
        area, perimeter, eccentricity = 0.0, 0.0, 0.0
    features += [area, perimeter, eccentricity]

    #  Summary dict
    feature_summary = {
        "color": {
            "r_mean": color_vals[0]["mean"],
            "g_mean": color_vals[1]["mean"],
            "b_mean": color_vals[2]["mean"],
            "r_std":  color_vals[0]["std"],
            "g_std":  color_vals[1]["std"],
            "b_std":  color_vals[2]["std"],
        },
        "histogram": {
            "r": hist_vals[0],
            "g": hist_vals[1],
            "b": hist_vals[2],
        },
        "glcm": glcm_vals,
        "lbp": {
            "uniformity": lbp_uniformity,
            "entropy":    lbp_entropy,
            "bins":       lbp_hist.tolist(),
        },
        "hog": {
            "mean":      hog_mean,
            "max":       hog_max,
            "magnitude": float(np.linalg.norm(hog_sub)),
        },
        "morphological": {
            "area":        float(area),
            "perimeter":   float(perimeter),
            "eccentricity": float(eccentricity),
        }
    }

    return np.array(features, dtype=np.float32), feature_summary