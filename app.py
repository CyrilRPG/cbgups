import io
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Lecture de grilles OUI/NON → Excel (A/B)", layout="wide")

# =========================================================
# ----------------------- UTILITAIRES ----------------------
# =========================================================

def load_bgr(file) -> np.ndarray:
    im = Image.open(file).convert("RGB")
    arr = np.array(im)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    # Upscale pour stabiliser la détection (scans basse résolution)
    bgr = cv2.resize(bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return bgr

def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def adaptive_bin(gray: np.ndarray, block_size=35, C=10) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, int(block_size) | 1, int(C)
    )

def find_checkbox_contours(
    bin_img: np.ndarray,
    min_area=120,
    max_area=6000,
    squareness_tol=0.40
) -> List[Tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = bin_img.shape[:2]
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area or area > max_area:
            continue
        # Carré ±
        ratio = abs(w - h) / max(w, h)
        if ratio > squareness_tol:
            continue
        # éviter les bords
        if x < 1 or y < 1 or x + w > W - 1 or y + h > H - 1:
            continue
        boxes.append((x, y, w, h))
    return boxes

def inner_roi(gray: np.ndarray, box: Tuple[int, int, int, int], margin: float = 0.18) -> np.ndarray:
    x, y, w, h = box
    mx, my = int(w * margin), int(h * margin)
    return gray[y + my: y + h - my, x + mx: x + w - mx]

def mark_score(roi_gray: np.ndarray) -> float:
    """
    Score 0..1 indiquant si la case est cochée (noircissage ou croix).
    Combine:
      - 1 - mean_intensity/255
      - proportion sombre (Otsu local)
      - présence de lignes (croix) via Hough
    """
    if roi_gray.size == 0:
        return 0.0

    mean_dark = 1.0 - (roi_gray.mean() / 255.0)

    # densité sombre relative
    _, thr = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_ratio = float((thr == 0).sum()) / float(roi_gray.size)

    # croix (2 segments ou plus)
    edges = cv2.Canny(roi_gray, 40, 120, L2gradient=True)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=10,
        minLineLength=max(4, roi_gray.shape[1] // 3),
        maxLineGap=4
    )
    line_bonus = 0.18 if (lines is not None and len(lines) >= 2) else 0.0

    score = 0.55 * mean_dark + 0.45 * dark_ratio + line_bonus
    return float(max(0.0, min(1.0, score)))

def group_into_rows(boxes: List[Tuple[int, int, int, int]], y_tol: int = 12) -> List[List[Tuple[int, int, int, int]]]:
    if not boxes:
        return []
    boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0]))
    rows, current = [], [boxes_sorted[0]]
    for b in boxes_sorted[1:]:
        if abs(b[1] - current[-1][1]) <= y_tol:
            current.append(b)
        else:
            rows.append(current)
            current = [b]
    if current:
        rows.append(current)
    rows = [sorted(r, key=lambda b: b[0]) for r in rows]
    return rows

def pair_by_proximity(
    row: List[Tuple[int, int, int, int]],
    x_gap_tol: int = 14
) -> List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
    pairs = []
    i = 0
    while i < len(row) - 1:
        b1, b2 = row[i], row[i + 1]
        gap = b2[0] - (b1[0] + b1[2])
        if gap <= max(x_gap_tol, int(0.7 * b1[2])):
            pairs.append((b1, b2))
            i += 2
        else:
            i += 1
    return pairs

def decide_AB(
    gray: np.ndarray,
    pair: Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]],
    thresh: float
) -> Tuple[str, float]:
    left, right = pair
    # scores robustes
    s_left = mark_score(inner_roi(gray, left))
    s_right = mark_score(inner_roi(gray, right))

    # logique : exactement une case au-dessus du seuil
    if (s_left >= thresh) ^ (s_right >= thresh):
        if s_left > s_right:
            return "A", float(s_left - s_right)
        else:
            return "B", float(s_right - s_left)

    # en cas d'ambiguïté ou aucune : vide
    return "", float(abs(s_left - s_right))

def draw_debug(
    img: np.ndarray,
    pairs: List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]],
    results: List[str]
) -> np.ndarray:
    vis = img.copy()
    for (left, right), r in zip(pairs, results):
        colL = colR = (180, 180, 180)  # gris (NA)
        if r == "A":
            colL, colR = (0, 200, 0), (70, 70, 70)
        elif r == "B":
            colL, colR = (70, 70, 70), (0, 0, 220)
        for (x, y, w, h), col in [(left, colL), (right, colR)]:
            cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
    return vis

def split_into_5_columns(
    pairs: List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]
) -> List[List]:
    """
    Sépare les paires en 5 colonnes par grandes coupures en X.
    Retourne [col1, col2, col3, col4, col5] (une éventuelle 6e est ignorée).
    """
    if not pairs:
        return [[], [], [], [], []]
    centers_x = np.array([(p[0][0] + p[1][0]) / 2.0 for p in pairs])
    order = np.argsort(centers_x)
    sx = centers_x[order]

    gaps = np.diff(sx)
    if len(gaps) < 4:
        # fallback : 5 tranches égales
        cuts = np.linspace(0, len(order), 6, dtype=int)
    else:
        top_idx = np.argsort(gaps)[-4:]
        cut_positions = np.sort(top_idx + 1)
        cuts = np.concatenate(([0], cut_positions, [len(order)]))

    cols_idx = [order[cuts[i]:cuts[i + 1]].tolist() for i in range(5)]
    columns = [[pairs[k] for k in idxs] for idxs in cols_idx]
    return columns

def sort_25_rows_per_column(columns: List[List], expected_rows=25) -> List[Tuple]:
    """
    Trie chaque colonne verticalement et tronque/complète à expected_rows lignes.
    """
    normalized = []
    for col in columns[:5]:
        col_sorted = sorted(col, key=lambda p: p[0][1])  # y du left
        if len(col_sorted) >= expected_rows:
            col_sorted = col_sorted[:expected_rows]
        else:
            pad = expected_rows - len(col_sorted)
            col_sorted += [None] * pad
        normalized.append(col_sorted)
    return normalized

# =========================================================
# ----------------------- INTERFACE -----------------------
# =========================================================

st.title("Lecture de grilles OUI/NON → Excel (A/B)")

with st.sidebar:
    st.header("Paramètres")
    expected_questions = st.number_input("Nombre de questions", min_value=1, value=125, step=1)
    questions_per_col = st.number_input("Questions par colonne", min_value=1, max_value=50, value=25, step=1)

    # Seuil de marquage
    thresh = st.slider("Seuil de marquage (0–1)", min_value=0.05, max_value=0.80, value=0.22, step=0.01)

    # Détection de contours : bornes VALIDES
    min_area = st.number_input("Aire min. case", min_value=10, max_value=10000, value=120, step=10)
    max_area = st.number_input("Aire max. case", min_value=200, max_value=30000, value=6000, step=50)
    squareness_tol = st.slider("Tolérance carré", min_value=0.0, max_value=0.8, value=0.40, step=0.01)

    # Groupements
    y_tol = st.number_input("Tolérance verticale (lignes)", min_value=2, max_value=60, value=12, step=1)
    x_gap_tol = st.number_input("Tolérance horizontale (paires)", min_value=2, max_value=80, value=14, step=1)

uploaded = st.file_uploader("Dépose une image (JPG/PNG)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img_bgr = load_bgr(uploaded)
    gray = to_gray(img_bgr)
    bin_img = adaptive_bin(gray)

    # 1) Détection des cases
    boxes = find_checkbox_contours(
        bin_img,
        min_area=int(min_area),
        max_area=int(max_area),
        squareness_tol=float(squareness_tol),
    )
    rows = group_into_rows(boxes, y_tol=int(y_tol))

    # 2) Paires OUI/NON par ligne
    all_pairs = []
    for row in rows:
        all_pairs.extend(pair_by_proximity(row, x_gap_tol=int(x_gap_tol)))

    # 3) Forcer la structure 5 colonnes × N (écarte tout surplus)
    cols = split_into_5_columns(all_pairs)
    cols = sort_25_rows_per_column(cols, expected_rows=int(questions_per_col))
    pairs_5xN = [p for col in cols[:5] for p in col if p is not None]

    # 4) Décision A/B/"" question par question
    results, confidences = [], []
    for pair in pairs_5xN:
        r, c = decide_AB(gray, pair, thresh=float(thresh))
        results.append(r)
        confidences.append(c)

    # 5) Tronquer/padder à expected_questions (sécurité)
    if len(results) >= expected_questions:
        results = results[:expected_questions]
        confidences = confidences[:expected_questions]
    else:
        pad = expected_questions - len(results)
        results += [""] * pad
        confidences += [0.0] * pad

    # 6) DataFrame EXACTEMENT comme ton modèle : Q1..Qn, 2e ligne = A/B/"".
    cols_names = [f"Q{i}" for i in range(1, expected_questions + 1)]
    df_ab = pd.DataFrame([[*results]], columns=cols_names)
    df_conf = pd.DataFrame([[round(c, 3) for c in confidences]], columns=cols_names)

    # 7) Aperçu + export Excel
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.subheader("Cases détectées (vert=A, rouge=B, gris=NA)")
        debug = draw_debug(img_bgr, pairs_5xN[:expected_questions], results[:expected_questions])
        st.image(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption(
            f"Paires détectées: {len(all_pairs)} | "
            f"Utilisées (5×{questions_per_col}): {len(pairs_5xN)}"
        )
    with c2:
        st.subheader("Aperçu Excel (A/B)")
        st.dataframe(df_ab, use_container_width=True, height=180)
        with st.expander("Confiance (0–1)"):
            st.dataframe(df_conf, use_container_width=True, height=140)

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df_ab.to_excel(w, index=False, sheet_name="Réponses")
    st.download_button(
        "⬇️ Télécharger l’Excel (A/B)",
        data=out.getvalue(),
        file_name="reponses_AB.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

else:
    st.info("Charge une image de grille pour lancer l’analyse.")
