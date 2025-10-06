# -*- coding: utf-8 -*-
# src/color_corrector.py
from pathlib import Path
import numpy as np, cv2, torch, torchvision
from PIL import Image
from colour import delta_E
from sklearn.linear_model import LogisticRegression
import ot

# -------------------------
# Utility conversions
# -------------------------
def to_lab(bgr):      # BGR uint8 -> Lab float
    return cv2.cvtColor((bgr.astype(np.float32)/255.0), cv2.COLOR_BGR2LAB)

def to_bgr(lab):     # Lab float -> BGR uint8
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2BGR)*255.0,0,255).astype(np.uint8)

def feather(mask_u8, sigma=2):
    m=(mask_u8>0).astype(np.float32)
    soft=cv2.GaussianBlur(m,(0,0),sigma,sigma); soft*=m
    if soft.max()>0: soft/=soft.max()
    return soft

def masked_de2000(a_bgr,b_bgr,mask_u8):
    LabA,LabB=to_lab(a_bgr),to_lab(b_bgr); m=mask_u8>0
    if m.sum()==0: return 0.0
    return float(delta_E(LabA[m],LabB[m], method="CIE 2000").mean())

# -------------------------
# Person seg / skin / face
# -------------------------
class PersonSeg:
    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(
            weights="COCO_WITH_VOC_LABELS_V1").to(self.device).eval()
        self.tfm = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    @torch.inference_mode()
    def person_mask(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = self.tfm(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        out = self.model(inp)["out"].softmax(1)[0].argmax(0).cpu().numpy().astype(np.uint8)
        return (out==15).astype(np.uint8)*255  # "person"

def skin_mask_dual(bgr):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    skin1 = cv2.inRange(ycrcb, (0,133,77), (255,173,127))
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    skin2 = cv2.inRange(hsv, (0,30,60), (25,180,255)) | cv2.inRange(hsv, (160,30,60), (179,180,255))
    skin  = skin1 | skin2
    skin  = cv2.dilate(skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),1)
    return skin

def face_bbox(bgr):
    modelFile   = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade= cv2.CascadeClassifier(modelFile)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces)==0: return None
    x,y,w,h = max(faces, key=lambda f:f[2]*f[3])
    return (x,y,w,h)

# -------------------------
# Still-life mask (auto if missing)
# -------------------------
def still_mask_autogen(still_bgr: np.ndarray) -> np.ndarray:
    """Quick GrabCut-from-center if still_mask.png is missing."""
    H,W = still_bgr.shape[:2]
    rect = (int(W*0.1), int(H*0.1), int(W*0.8), int(H*0.8))
    GC_BGD,GC_FGD,GC_PR_BGD,GC_PR_FGD = 0,1,2,3
    mask = np.zeros((H,W), np.uint8)
    bgdModel = np.zeros((1,65),np.float64); fgdModel=np.zeros((1,65),np.float64)
    cv2.grabCut(still_bgr, mask, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)
    m = np.where((mask==GC_FGD)|(mask==GC_PR_FGD),255,0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),1)
    return m

# -------------------------
# On-model mask (v3) + fallback (colour classifier)
# -------------------------
def ab_prior_from_still(still_bgr, still_mask_u8):
    lab = to_lab(still_bgr)
    ab  = lab[...,1:][still_mask_u8>0]
    mu = ab.mean(0); sd = ab.std(0)+1e-3
    return mu, sd

def onmodel_garment_mask_v3(om_bgr, still_bgr, still_mask_u8, person_seg: PersonSeg):
    H,W = om_bgr.shape[:2]
    pm  = person_seg.person_mask(om_bgr)
    if pm.sum()<500: return np.zeros((H,W), np.uint8)
    skin = skin_mask_dual(om_bgr)

    torso = (pm>0).astype(np.uint8)*255
    torso[skin>0] = 0

    fb = face_bbox(om_bgr)
    y_top_guess = int(H*0.18)
    if fb:
        x,y,w,h = fb
        y_top_guess = int(min(H, y + h*1.05))

    ys = np.where(torso>0)[0]
    if ys.size==0: return np.zeros((H,W), np.uint8)
    y_min, y_max = ys.min(), ys.max()

    band_top = max(y_top_guess, y_min)
    band_bot = int(y_min + 0.70*(y_max - y_min))
    band_bot = min(band_bot, H-1)
    band = np.zeros((H,W), np.uint8); band[band_top:band_bot,:]=255

    seed0 = torso & band

    mu, sd = ab_prior_from_still(still_bgr, still_mask_u8)
    lab = to_lab(om_bgr); ab = lab[...,1:]
    z = np.sqrt(((ab - mu)**2 / (sd**2)).sum(2))

    extend_rows = int(0.10*(y_max - y_min))
    y2 = min(H-1, band_bot+extend_rows)
    cand = np.zeros((H,W), np.uint8); cand[band_bot:y2,:]=255
    close = (z<2.2).astype(np.uint8)*255
    extra = cand & close & (pm>0).astype(np.uint8)*255 & (skin==0).astype(np.uint8)*255
    seed = cv2.morphologyEx(seed0 | extra, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),1)

    GC_BGD,GC_FGD,GC_PR_BGD,GC_PR_FGD = 0,1,2,3
    seeds = np.full((H,W), GC_PR_BGD, np.uint8)

    sure_fg = (seed>0) & (z<2.0) & (skin==0) & (pm>0)
    sure_bg = ((pm==0) | (skin>0) | (z>4.5))

    MIN_PIX = 200
    if sure_fg.sum() < MIN_PIX:
        sure_fg = (seed>0) & (z<2.6) & (skin==0) & (pm>0)
        sure_fg = cv2.dilate(sure_fg.astype(np.uint8)*255,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),1)>0
    if sure_bg.sum() < MIN_PIX:
        ring = np.zeros((H,W), np.uint8)
        ring[:8,:]=255; ring[-8:,:]=255; ring[:,:8]=255; ring[:,-8:]=255
        sure_bg = sure_bg | (ring>0)

    seeds[sure_fg] = GC_FGD
    seeds[sure_bg] = GC_BGD

    if (seeds==GC_FGD).sum() < MIN_PIX or (seeds==GC_BGD).sum() < MIN_PIX:
        x0, x1 = int(W*0.15), int(W*0.85)
        y0, y1 = max(0, band_top-10), min(H-1, band_bot+10)
        rect = (x0, y0, max(10,x1-x0), max(10,y1-y0))
        mask_gc = np.zeros((H,W), np.uint8)
        bgdModel = np.zeros((1,65),np.float64); fgdModel=np.zeros((1,65),np.float64)
        cv2.grabCut(om_bgr, mask_gc, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask_gc==GC_FGD)|(mask_gc==GC_PR_FGD),255,0).astype(np.uint8)
    else:
        bgdModel = np.zeros((1,65),np.float64); fgdModel=np.zeros((1,65),np.float64)
        cv2.grabCut(om_bgr, seeds, None, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
        mask = np.where((seeds==GC_FGD)|(seeds==GC_PR_FGD), 255, 0).astype(np.uint8)

    cnts,_ = cv2.findContours((mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = np.zeros((H,W), np.uint8)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(keep,[c],-1,255,cv2.FILLED)
        keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),1)
    return keep

def onmodel_garment_mask_colorclf(om_bgr, still_bgr, still_mask_u8, person_seg: PersonSeg):
    H,W = om_bgr.shape[:2]
    pm  = person_seg.person_mask(om_bgr)
    if pm.sum()<500: return np.zeros((H,W), np.uint8)
    skin = skin_mask_dual(om_bgr)

    fb = face_bbox(om_bgr)
    y_top_guess = int(H*0.18)
    if fb:
        x,y,w,h = fb
        y_top_guess = int(min(H, y + h*1.05))
    ys = np.where(pm>0)[0]
    if ys.size==0: return np.zeros((H,W), np.uint8)
    y_min, y_max = ys.min(), ys.max()
    band_top = max(y_top_guess, y_min)
    band_bot = int(y_min + 0.70*(y_max - y_min))
    band_bot = min(band_bot, H-1)
    band = np.zeros((H,W), np.uint8); band[band_top:band_bot,:]=255

    lab_s = to_lab(still_bgr); ab_pos = lab_s[...,1:][still_mask_u8>0]
    lab_o = to_lab(om_bgr);    ab_all = lab_o[...,1:]
    bg_mask = (pm==0)
    ab_neg = ab_all[bg_mask][:min(5000, bg_mask.sum())]
    if ab_pos.shape[0] < 100 or ab_neg.shape[0] < 100:
        return onmodel_garment_mask_v3(om_bgr, still_bgr, still_mask_u8, person_seg)

    X = np.vstack([ab_pos, ab_neg]).astype(np.float32)
    y = np.hstack([np.ones(len(ab_pos)), np.zeros(len(ab_neg))])
    clf = LogisticRegression(max_iter=200, class_weight='balanced').fit(X, y)

    idx = np.where((band>0) & (skin==0))
    proba = np.zeros((H,W), np.float32)
    if len(idx[0])>0:
        proba[idx] = clf.predict_proba(ab_all[idx])[:,1]

    GC_BGD,GC_FGD,GC_PR_BGD,GC_PR_FGD = 0,1,2,3
    seeds = np.full((H,W), GC_PR_BGD, np.uint8)
    seeds[(proba>=0.7)] = GC_FGD
    seeds[(proba<=0.2)] = GC_BGD
    seeds[(pm==0)]      = GC_BGD
    seeds[(skin>0)]     = GC_BGD

    bgdModel = np.zeros((1,65),np.float64); fgdModel=np.zeros((1,65),np.float64)
    cv2.grabCut(om_bgr, seeds, None, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
    mask = np.where((seeds==GC_FGD)|(seeds==GC_PR_FGD), 255, 0).astype(np.uint8)

    cnts,_ = cv2.findContours((mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = np.zeros((H,W), np.uint8)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(keep,[c],-1,255,cv2.FILLED)
        keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),1)
    return keep

def build_onmodel_mask_with_fallback(om_bgr, sl_bgr, sm_u8, person_seg: PersonSeg):
    mm = onmodel_garment_mask_v3(om_bgr, sl_bgr, sm_u8, person_seg)
    if (mm>0).sum() < 1200:
        mm2 = onmodel_garment_mask_colorclf(om_bgr, sl_bgr, sm_u8, person_seg)
        if (mm2>0).sum() > (mm>0).sum():
            return mm2
    return mm

# -------------------------
# Degradation (smart)
# -------------------------
def degrade_smart(om_bgr, on_mask_u8, seed=None,
                  target_de=5.0, max_tries=8,
                  chroma_thresh=8.0, boost_range=(12,20)):
    rng = np.random.default_rng(seed)
    m_full = (on_mask_u8 > 0)
    if m_full.sum() == 0:
        return om_bgr.copy()

    m_erode = cv2.erode(on_mask_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1) > 0

    def lab_radial_boost(bgr, mask, target_radius):
        lab = to_lab(bgr)
        a = lab[...,1]; b = lab[...,2]
        a_m = a[mask]; b_m = b[mask]
        r   = np.sqrt(a_m**2 + b_m**2) + 1e-6
        a_t = a_m * (target_radius / r)
        b_t = b_m * (target_radius / r)
        lab[...,1][mask] = np.clip(a_t, -128, 127)
        lab[...,2][mask] = np.clip(b_t, -128, 127)
        return to_bgr(lab)

    def hsv_jitter(bgr, mask, h_deg=(10,25), s_mul=(0.85,1.25), v_mul=(0.92,1.10)):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_shift = rng.choice([-1,1]) * rng.uniform(*h_deg)
        s_scale = rng.uniform(*s_mul)
        v_scale = rng.uniform(*v_mul)
        hsv[...,0][mask] = (hsv[...,0][mask] + h_shift/2) % 180
        hsv[...,1][mask] = np.clip(hsv[...,1][mask]*s_scale, 0,255)
        hsv[...,2][mask] = np.clip(hsv[...,2][mask]*v_scale, 0,255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    base = om_bgr.copy()
    lab0 = to_lab(base); ab0 = lab0[...,1:]
    mean_chroma = float(np.linalg.norm(ab0[m_full], axis=1).mean()) if m_full.any() else 0.0

    out = base.copy()
    for t in range(max_tries):
        x = out.copy()
        if mean_chroma < chroma_thresh:
            target_r = np.random.uniform(*boost_range) * (1.0 + 0.12*t)
            x = lab_radial_boost(x, m_erode, target_r)
        h_step = (10, 25+2*t)
        s_step = (0.85, min(1.35, 1.25+0.03*t))
        v_step = (0.92, min(1.15, 1.10+0.02*t))
        x = hsv_jitter(x, m_erode, h_deg=h_step, s_mul=s_step, v_mul=v_step)

        de = masked_de2000(base, x, on_mask_u8)
        if de >= target_de:
            return x
        out = x
    return out

# simple alias for compatibility
def ensure_visible_degrade(om_bgr, mask_u8, min_de=4.5, max_tries=6):
    return degrade_smart(om_bgr, mask_u8, target_de=min_de, max_tries=max_tries)

# -------------------------
# Correction (Reinhard a*b* + cap; optional OT blend)
# -------------------------
def reinhard_ab(src_lab, ref_lab, src_mask, ref_mask):
    ms = src_mask>0; mr = ref_mask>0
    if ms.sum()==0 or mr.sum()==0: return src_lab
    mu_s = src_lab[ms].mean(0);  sd_s = src_lab[ms].std(0)+1e-6
    mu_r = ref_lab[mr].mean(0);  sd_r = ref_lab[mr].std(0)+1e-6
    out  = src_lab.copy()
    out[ms,1:] = ((out[ms,1:]-mu_s[1:])/sd_s[1:])*sd_r[1:] + mu_r[1:]
    out[...,1] = np.clip(out[...,1], -128,127)
    out[...,2] = np.clip(out[...,2], -128,127)
    return out

def cap_ab_delta(orig_ab, new_ab, max_norm=11.0):
    delta = new_ab - orig_ab
    n = np.linalg.norm(delta, axis=1, keepdims=True)+1e-6
    scale = np.minimum(1.0, max_norm/n)
    return orig_ab + delta*scale

def build_ab_hist(ab_vals, bin_size=8):
    a_edges=np.arange(-128,128+1e-3,bin_size)
    b_edges=np.arange(-128,128+1e-3,bin_size)
    H,_,_ = np.histogram2d(ab_vals[:,0], ab_vals[:,1], bins=[a_edges,b_edges])
    a_cent=(a_edges[:-1]+a_edges[1:])/2; b_cent=(b_edges[:-1]+b_edges[1:])/2
    A,B=np.meshgrid(a_cent,b_cent,indexing="ij")
    centers=np.stack([A.ravel(),B.ravel()],1)
    return H.ravel().astype(np.float64), centers, (a_edges,b_edges)

def sinkhorn_map(src_ab, ref_ab, reg=0.05, bin_size=8, maxiter=300, eps=1e-8):
    h_s, centers, (a_edges,b_edges) = build_ab_hist(src_ab, bin_size)
    h_t,_,_ = build_ab_hist(ref_ab, bin_size)
    if h_s.sum()==0 or h_t.sum()==0:
        return lambda ab: ab
    a = (h_s + eps); b = (h_t + eps)
    a = a / a.sum(); b = b / b.sum()
    C = ot.dist(centers, centers, metric='euclidean')**2
    P = ot.sinkhorn(a, b, C, reg=reg, numItermax=maxiter, stopThr=1e-6, warn=False)
    denom = P.sum(1,keepdims=True)+1e-12
    mapped=(P@centers)/denom
    def apply_fn(ab):
        ai=np.clip(np.digitize(ab[:,0],a_edges)-1,0,len(a_edges)-2)
        bi=np.clip(np.digitize(ab[:,1],b_edges)-1,0,len(b_edges)-2)
        idx=ai*(len(b_edges)-1)+bi
        return mapped[idx].astype(np.float32)
    return apply_fn

def correct_one(still_bgr, still_mask_u8, on_deg_bgr, on_orig_bgr, on_mask_u8, use_ot=False):
    ref = to_lab(still_bgr)
    deg = to_lab(on_deg_bgr)
    ori = to_lab(on_orig_bgr)

    warm = reinhard_ab(deg, ref, on_mask_u8, still_mask_u8)

    m = (on_mask_u8>0).ravel()
    ab0 = deg[...,1:].reshape(-1,2)
    ab1 = warm[...,1:].reshape(-1,2)
    ab1[m] = cap_ab_delta(ab0[m], ab1[m], max_norm=11.0)
    warm[...,1:] = ab1.reshape(deg.shape[0],deg.shape[1],2)

    if use_ot:
        ref_ab = ref[...,1:][still_mask_u8>0].reshape(-1,2)
        out_ab = warm[...,1:][on_mask_u8>0].reshape(-1,2)
        if ref_ab.size and out_ab.size:
            de = np.linalg.norm(ref_ab.mean(0)-out_ab.mean(0))
            if de > 8.0:
                apply = sinkhorn_map(out_ab, ref_ab, reg=0.05, bin_size=8, maxiter=250)
                mapped = apply(out_ab)
                ab_blend = 0.30*mapped + 0.70*out_ab
                flat = warm[...,1:].reshape(-1,2)
                flat[m] = ab_blend
                warm[...,1:] = flat.reshape(deg.shape[0],deg.shape[1],2)

    warm[...,0] = ori[...,0]
    base = deg.copy(); base[...,0] = ori[...,0]

    w = feather(on_mask_u8, sigma=2)[...,None]
    lab_out = warm*w + base*(1.0-w)
    return to_bgr(lab_out)
