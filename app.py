import io
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Lecture de grilles OUI/NON → Excel A/B", layout="wide")


# =========================================================
# --------------------- UTILITAIRES -----------------------
# =========================================================

def load_bgr(file) -> np.ndarray:
    image = Image.open(file).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def adaptive_bin(gray: np.ndarray, block_size=35, C=10) -> np.ndarray:
    """Binarisation robuste aux éclairages inégaux, inversion (encre = blanc)."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size | 1, C
    )

def find_checkbox_contours(bin_img: np.ndarray,
                           min_area=120, max_area=3500,
                           squareness_tol=0.35) -> List[Tuple[int,int,int,int]]:
    """
    Repère les petits carrés (cases) via leurs contours.
    squareness_tol = tolérance sur le rapport L/H (0 = carré parfait).
    """
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    H, W = bin_img.shape[:2]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area or area > max_area:
            continue
        ratio = abs(w - h) / max(w, h)
        if ratio > squareness_tol:
            continue
        # rejeter les bords de page
        if x < 5 or y < 5 or x + w > W - 5 or y + h > H - 5:
            continue
        boxes.append((x, y, w, h))
    return boxes

def inner_roi(gray: np.ndarray, box: Tuple[int,int,int,int], margin: float = 0.22) -> np.ndarray:
    x, y, w, h = box
    mx, my = int(w * margin), int(h * margin)
    return gray[y + my: y + h - my, x + mx: x + w - mx]

def mark_score(roi_gray: np.ndarray) -> float:
    """
    Score de "remplissage" d'une case (0..1).
    Combine densité sombre + traits (croix) potentiels.
    """
    if roi_gray.size == 0:
        return 0.0

    # Densité sombre relative (Otsu local)
    _, thr = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = (thr == 0).sum() / float(roi_gray.size)

    # Détection de traits (croix possible)
    edges = cv2.Canny(roi_gray, 50, 150, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=12, minLineLength=max(4, roi_gray.shape[1]//3), maxLineGap=3)
    line_bonus = 0.15 if lines is not None and len(lines) >= 2 else 0.0

    return min(1.0, dark + line_bonus)

def group_into_rows(boxes: List[Tuple[int,int,int,int]], y_tol: int = 10) -> List[List[Tuple[int,int,int,int]]]:
    """
    Groupe les cases en lignes par proximité verticale.
    """
    if not boxes:
        return []
    boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0]))
    rows = []
    current = [boxes_sorted[0]]
    for b in boxes_sorted[1:]:
        if abs(b[1] - current[-1][1]) <= y_tol:
            current.append(b)
        else:
            rows.append(current)
            current = [b]
    if current:
        rows.append(current)
    # trier horizontalement dans chaque ligne
    rows = [sorted(r, key=lambda b: b[0]) for r in rows]
    return rows

def pair_by_proximity(row: List[Tuple[int,int,int,int]], x_gap_tol: int = 12) -> List[Tuple[Tuple[int,int,int,int], Tuple[int,int,int,int]]]:
    """
    Construit des paires (gauche, droite) dans une ligne.
    """
    pairs = []
    i = 0
    while i < len(row) - 1:
        b1, b2 = row[i], row[i+1]
        # on suppose que les cases viennent en paires rapprochées
        if (b2[0] - (b1[0] + b1[2])) <= max(x_gap_tol, int(0.6 * b1[2])):
            pairs.append((b1, b2))
            i += 2
        else:
            # élément isolé : l'ignorer pour rester robuste
            i += 1
    return pairs

def decide_AB(gray: np.ndarray,
              pair: Tuple[Tuple[int,int,int,int], Tuple[int,int,int,int]],
              thresh: float) -> Tuple[str, float]:
    """
    Retourne ("A" | "B" | "", confiance).
    A = OUI (case gauche), B = NON (case droite), "" = non répondu / ambigu.
    """
    left, right = pair
    s_left = mark_score(inner_roi(gray, left))
    s_right = mark_score(inner_roi(gray, right))

    # exactement une case "clairement" marquée
    if (s_left >= thresh) ^ (s_right >= thresh):
        if s_left > s_right:
            conf = s_left - s_right
            return "A", float(conf)
        else:
            conf = s_right - s_left
            return "B", float(conf)

    # aucune dépassant le seuil -> vide (non répondu)
    if s_left < thresh and s_right < thresh:
        return "", float(max(s_left, s_right))

    # les deux dépassent -> ambigu -> vide
    return "", float(abs(s_left - s_right))

def draw_debug(img: np.ndarray,
               pairs: List[Tuple[Tuple[int,int,int,int], Tuple[int,int,int,int]]],
               results: List[str]) -> np.ndarray:
    vis = img.copy()
    for (left, right), r in zip(pairs, results):
        color_left = (128, 128, 128)
        color_right = (128, 128, 128)
        if r == "A":
            color_left = (0, 200, 0)   # vert
            color_right = (60, 60, 60)
        elif r == "B":
            color_left = (60, 60, 60)
            color_right = (0, 0, 200)  # rouge
        else:
            color_left = color_right = (180, 180, 180)  # gris

        for (x, y, w, h), col in [(left, color_left), (right, color_right)]:
            cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
    return vis


# =========================================================
# ---------------------- INTERFACE ------------------------
# =========================================================

st.title("Lecture de grilles OUI/NON → Excel (A/B)")

with st.sidebar:
    st.header("Paramètres")
    expected_questions = st.number_input("Nombre de questions attendues", min_value=1, value=145, step=1)
    thresh = st.slider("Seuil de marquage (0-1)", 0.10, 0.80, 0.33, 0.01)
    min_area = st.number_input("Aire min. d'une case", 50, 5000, 120, 10)
    max_area = st.number_input("Aire max. d'une case", 100, 20000, 3500, 50)
    squareness_tol = st.slider("Tolérance carré (0=parfait)", 0.0, 0.8, 0.35, 0.01)
    y_tol = st.number_input("Tolérance verticale (groupes en lignes)", 2, 40, 10, 1)
    x_gap_tol = st.number_input("Tolérance horizontale (paires)", 2, 60, 12, 1)

uploaded = st.file_uploader("Dépose une image de grille (JPG/PNG/PDF¹)", type=["png", "jpg", "jpeg"])
st.caption("¹ Pour un PDF, exporte-le en image avant import si besoin.")

if uploaded is not None:
    img_bgr = load_bgr(uploaded)
    gray = to_gray(img_bgr)
    bin_img = adaptive_bin(gray)

    # Détection cases -> lignes -> paires
    boxes = find_checkbox_contours(bin_img, min_area=min_area, max_area=max_area, squareness_tol=squareness_tol)
    rows = group_into_rows(boxes, y_tol=int(y_tol))

    all_pairs = []
    for row in rows:
        all_pairs.extend(pair_by_proximity(row, x_gap_tol=int(x_gap_tol)))

    # Ordonner globalement top->bottom by y, puis left->right
    all_pairs = sorted(all_pairs, key=lambda p: (p[0][1], p[0][0]))

    # Prise de décision A/B/""
    results = []
    confidences = []
    for pair in all_pairs:
        r, c = decide_AB(gray, pair, thresh=thresh)
        results.append(r)
        confidences.append(c)

    # Tronquer/padder jusqu'au nombre de questions attendu
    if len(results) >= expected_questions:
        results = results[:expected_questions]
        confidences = confidences[:expected_questions]
    else:
        pad = expected_questions - len(results)
        results += [""] * pad
        confidences += [0.0] * pad

    # Construire DataFrame EXACTEMENT comme voulu : colonnes Q1..Qn, 1 seule ligne avec A/B/"".
    cols = [f"Q{i}" for i in range(1, expected_questions + 1)]
    df_row = {f"Q{i+1}": results[i] for i in range(expected_questions)}
    df_ab = pd.DataFrame([df_row], columns=cols)

    # Optionnel : ligne de confiance (utile au débogage, non exportée par défaut)
    df_conf = pd.DataFrame([ {f"Q{i+1}": round(confidences[i], 3) for i in range(expected_questions)} ])

    # Prévisualisations
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.subheader("Image et cases détectées")
        debug = draw_debug(img_bgr, all_pairs[:expected_questions], results[:expected_questions])
        st.image(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB), use_column_width=True)
    with c2:
        st.subheader("Aperçu Excel (A/B)")
        st.dataframe(df_ab, use_container_width=True, height=180)
        with st.expander("Confiance (0-1) par question"):
            st.dataframe(df_conf, use_container_width=True, height=140)

    # Export Excel au format exact : en-têtes Q1..Qn, deuxième ligne = A/B/""
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_ab.to_excel(writer, index=False, sheet_name="Réponses")
    st.download_button(
        "⬇️ Télécharger l’Excel (A/B)",
        data=out.getvalue(),
        file_name="reponses_AB.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

else:
    st.info("Charge une image de grille pour lancer l’analyse.")


# =========================================================
# ----------------------- NOTES ---------------------------
# - OUI = case gauche  → renvoie 'A' dans l’Excel
# - NON = case droite  → renvoie 'B' dans l’Excel
# - Aucune case marquée (ou double marquage) → cellule vide
# - Ajuste le seuil et les tolérances dans la barre latérale si besoin.
# =========================================================
