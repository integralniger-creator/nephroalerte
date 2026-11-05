import os
from pathlib import Path
import time
from datetime import date, datetime, timedelta
from io import BytesIO
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from reportlab.graphics.barcode import createBarcodeDrawing
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

try:
    import qrcode
except Exception:
    qrcode = None

try:
    import altair as alt
except Exception:
    alt = None


st.set_page_config(
    page_title="NéphroAlerte - Santé rénale",
    page_icon="Nephroalerte_logo.png",
    layout="wide",
)

with open(Path(__file__).with_name("nephroalerte_ui.css"), encoding="utf-8") as _ui_css_file:
    st.markdown(f"<style>{_ui_css_file.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    try:
        explainer = joblib.load("explainer.pkl")
    except Exception:
        classifier = model.named_steps["classifier"]
        explainer = shap.TreeExplainer(classifier)
    return model, explainer


model, explainer = load_assets()

FEATURE_MAPPING = {
    "num__creatininemie_mg_dl": "Créatininémie",
    "num__diurese_ml_24h": "Diurèse (24h)",
    "num__azotemie_mg_dl": "Azotémie",
    "num__pa_systolique": "Pression artérielle systolique",
    "num__pa_diastolique": "Pression artérielle diastolique",
    "num__age": "Âge",
    "num__hb_g_dl": "Hémoglobine",
    "num__gb_10e3_ul": "Globules blancs",
    "num__plaquettes_10e3_ul": "Plaquettes",
    "cat__hydratation_Normale": "Hydratation normale",
    "cat__hydratation_Déshydratation légère": "Déshydratation légère",
    "cat__hydratation_Déshydratation modérée": "Déshydratation modérée",
    "cat__hydratation_Déshydratation sévère": "Déshydratation sévère",
    "cat__hydratation_Hypovolémie": "Hypovolémie",
    "cat__hydratation_Œdèmes / surcharge": "Œdèmes / surcharge",
    "cat__antecedents_Aucun": "Absence d’antécédents",
    "cat__antecedents_IRC connue": "Insuffisance rénale chronique",
    "cat__antecedents_HTA": "Hypertension artérielle",
    "cat__antecedents_HTA + Diabète": "Hypertension + diabète",
    "cat__provenance_Rural": "Origine rurale",
    "cat__provenance_Urbain": "Origine urbaine",
    "cat__provenance_Périurbain": "Origine périurbaine",
}

PROVENANCES = ["Urbain", "Rural", "Périurbain"]
PROFESSIONS = ["Agriculteur", "Éleveur", "Commerçant", "Ménagère", "Fonctionnaire", "Ouvrier", "Transporteur", "Étudiant", "Sans emploi"]
ANTECEDENTS = ["Aucun", "HTA", "Diabète", "HTA + Diabète", "IRC connue", "Lithiase rénale", "Exposition néphrotoxique", "Infection récente"]
HYDRATATIONS = ["Normale", "Déshydratation légère", "Déshydratation modérée", "Déshydratation sévère", "Hypovolémie", "Œdèmes / surcharge"]

CLINIC_LOGO_SOURCE = "clinic_logo.png"
MINISTRY_LOGO_SOURCE = "ministry_logo.png"


def _read_source_bytes(source):
    if source is None:
        return None
    try:
        if hasattr(source, "read"):
            data = source.read()
            return data if isinstance(data, (bytes, bytearray)) else None
        if isinstance(source, (bytes, bytearray)):
            return bytes(source)
        if isinstance(source, BytesIO):
            return source.getvalue()
        if isinstance(source, str):
            if source.startswith("http://") or source.startswith("https://"):
                with urlopen(source) as response:
                    return response.read()
            if os.path.exists(source):
                with open(source, "rb") as f:
                    return f.read()
    except Exception:
        return None
    return None


def _image_reader(source):
    data = _read_source_bytes(source)
    if not data:
        return None
    return ImageReader(BytesIO(data))


def _rl_image(source, width, height):
    data = _read_source_bytes(source)
    if not data:
        return None
    return RLImage(BytesIO(data), width=width, height=height)


def _safe(v):
    return "Non renseigné" if v in (None, "", [], {}) else str(v)


def build_report_data(patient_meta, prediction, proba, shap_df):
    severity_map = {1: "Faible", 2: "Modéré", 3: "Élevé"}
    patient = {
        "numero_identification": _safe(patient_meta.get("numero_identification")),
        "nom": _safe(patient_meta.get("nom")),
        "prenom": _safe(patient_meta.get("prenom")),
        "telephone": _safe(patient_meta.get("telephone")),
        "adresse": _safe(patient_meta.get("adresse")),
        "date_naissance": _safe(patient_meta.get("date_naissance")),
        "lieu_naissance": _safe(patient_meta.get("lieu_naissance")),
        "photo": patient_meta.get("photo_patient"),
        "age": _safe(patient_meta.get("age")),
        "provenance": _safe(patient_meta.get("provenance")),
        "profession": _safe(patient_meta.get("profession")),
    }
    confidence = round(float(np.max(proba)) * 100, 1)
    factors = []
    if shap_df is not None and not shap_df.empty:
        for _, row in shap_df.head(5).iterrows():
            factors.append((FEATURE_MAPPING.get(row["feature"], row["feature"]), f"{float(row['impact']):.3f}"))
    return {
        "institution": {
            "republique": "République du Niger",
            "ministere": "Ministère de la Santé et de l’Hygiène Publique du Niger",
            "hopital": "Hôpital National de Zinder",
            "clinic_logo": CLINIC_LOGO_SOURCE,
            "ministry_logo": MINISTRY_LOGO_SOURCE,
        },
        "patient": patient,
        "analysis": {
            "prediction": int(prediction),
            "severity": severity_map.get(int(prediction), "Inconnu"),
            "confidence": confidence,
            "confidence_label": "Confiance élevée" if confidence >= 80 else "Confiance modérée" if confidence >= 60 else "Incertitude élevée",
            "factors": factors,
            "prevention": ["Hydratation", "Suivi régulier"] if int(prediction) == 1 else ["Réévaluation clinique", "Surveillance rapprochée", "Examens complémentaires"],
            "terrain": ["Vérifier l’accès à l’eau", "Évaluer l’exposition à la chaleur", "Sensibiliser le patient"],
        },
        "generated_at": datetime.now(),
    }


def _kv_table(title, rows, col_widths=(58 * mm, 102 * mm)):
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body_small", parent=styles["BodyText"], fontName="Helvetica", fontSize=8, leading=10, textColor=colors.HexColor("#153448"))
    data = [[Paragraph(title, ParagraphStyle("t", parent=styles["Heading4"], fontName="Helvetica-Bold", fontSize=9, leading=11, textColor=colors.white)), ""]]
    for a, b in rows:
        data.append([Paragraph(str(a), body), Paragraph(str(b), body)])
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("SPAN", (0, 0), (1, 0)),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#153448")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (1, 0), "CENTER"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#C9D9E5")),
        ("BOX", (0, 0), (-1, -1), 0.65, colors.HexColor("#AFC4D4")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FBFD")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl


def _bullet_table(title, items, col_width=160 * mm):
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body_small_2", parent=styles["BodyText"], fontName="Helvetica", fontSize=9, leading=11, textColor=colors.HexColor("#153448"))
    data = [[Paragraph(title, ParagraphStyle("t2", parent=styles["Heading4"], fontName="Helvetica-Bold", fontSize=10, leading=12, textColor=colors.white)), ""]]
    for item in items:
        data.append([Paragraph(str(item), body), ""])
    tbl = Table(data, colWidths=[col_width, 0.1])
    tbl.setStyle(TableStyle([
        ("SPAN", (0, 0), (1, 0)),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#153448")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (1, 0), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#C9D9E5")),
        ("BOX", (0, 0), (-1, -1), 0.65, colors.HexColor("#AFC4D4")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FBFD")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return tbl


def render_medical_sheet(report_data):
    st.markdown("""
        <style>
        .sheet-title { text-align:center; font-size:28px; font-weight:800; margin:10px 0 18px 0; color:#111827; }
        .sheet-subtitle { text-align:center; font-size:16px; font-weight:700; color:#111827; margin-top:2px; }
        .sheet-line { height:3px; background:#111827; margin:18px auto 18px auto; width:34%; }
        </style>
    """, unsafe_allow_html=True)
    inst = report_data["institution"]
    patient = report_data["patient"]
    analysis = report_data["analysis"]
    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        if os.path.exists(inst["clinic_logo"]):
            st.image(inst["clinic_logo"], width=110)
    with c2:
        st.markdown(f"<div class='sheet-subtitle'>{inst['republique']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='sheet-subtitle'>{inst['ministere']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='sheet-subtitle'>{inst['hopital']}</div>", unsafe_allow_html=True)
    with c3:
        if os.path.exists(inst["ministry_logo"]):
            st.image(inst["ministry_logo"], width=110)
    st.markdown("<div class='sheet-line'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sheet-title'>FICHE MEDICALE</div>", unsafe_allow_html=True)
    left, right = st.columns([1, 2])
    with left:
        if patient["photo"] is not None:
            st.image(patient["photo"], use_container_width=True)
        else:
            st.info("Photo patient non renseignée")
        qr_payload = f"{patient['nom']} | {patient['prenom']} | {report_data['generated_at'].strftime('%d/%m/%Y')} | KDIGO {analysis['prediction']}"
        if qrcode is not None:
            qr_image = qrcode.make(qr_payload).get_image()
            st.image(qr_image, width=170)
    with right:
        st.table(pd.DataFrame([
            ("N° ID", patient["numero_identification"]),
            ("Nom", patient["nom"]),
            ("Prénom", patient["prenom"]),
            ("Téléphone", patient["telephone"]),
            ("Adresse", patient["adresse"]),
            ("Date de naissance", patient["date_naissance"]),
            ("Lieu de naissance", patient["lieu_naissance"]),
        ], columns=["Champ", "Valeur"]))
    st.markdown("<div class='sheet-title' style='font-size:24px; margin-top:18px;'>INFORMATIONS PATIENT</div>", unsafe_allow_html=True)
    st.table(pd.DataFrame([
        ("Âge", patient["age"]),
        ("Provenance", patient["provenance"]),
        ("Profession", patient["profession"]),
    ], columns=["Champ", "Valeur"]))
    st.markdown("<div class='sheet-title' style='font-size:24px; margin-top:18px;'>RÉSULTATS</div>", unsafe_allow_html=True)
    st.table(pd.DataFrame([
        ("Classification KDIGO", analysis["prediction"]),
        ("Niveau de risque", analysis["severity"]),
        ("Niveau de confiance", f"{analysis['confidence']:.1f}%"),
        ("Appréciation", analysis["confidence_label"]),
    ], columns=["Champ", "Valeur"]))
    if analysis["factors"]:
        st.markdown("<div class='sheet-title' style='font-size:22px; margin-top:18px;'>INTERPRÉTATION CLINIQUE</div>", unsafe_allow_html=True)
        st.table(pd.DataFrame(analysis["factors"], columns=["Facteur principal", "Impact"]))
    st.markdown("<div class='sheet-title' style='font-size:22px; margin-top:18px;'>NIVEAU DE GRAVITÉ</div>", unsafe_allow_html=True)
    st.table(pd.DataFrame([("KDIGO", analysis["prediction"]), ("Niveau", analysis["severity"])], columns=["Champ", "Valeur"]))
    st.markdown("<div class='sheet-title' style='font-size:22px; margin-top:18px;'>PRÉVENTION</div>", unsafe_allow_html=True)
    st.table(pd.DataFrame([(x,) for x in analysis["prevention"]], columns=["Action"]))
    st.markdown("<div class='sheet-title' style='font-size:22px; margin-top:18px;'>CONTEXTE TERRAIN</div>", unsafe_allow_html=True)
    st.table(pd.DataFrame([(x,) for x in analysis["terrain"]], columns=["Action"]))


def _pdf_header(canvas, doc, report_data):
    inst = report_data["institution"]
    width, height = A4
    clinic_logo = _image_reader(inst["clinic_logo"])
    ministry_logo = _image_reader(inst["ministry_logo"])
    if clinic_logo:
        canvas.drawImage(clinic_logo, 14 * mm, height - 31 * mm, width=24 * mm, height=24 * mm, preserveAspectRatio=True, mask="auto")
    if ministry_logo:
        canvas.drawImage(ministry_logo, width - 38 * mm, height - 31 * mm, width=24 * mm, height=24 * mm, preserveAspectRatio=True, mask="auto")
    canvas.setFont("Helvetica-Bold", 18)
    canvas.setFillColor(colors.HexColor("#111111"))
    canvas.drawCentredString(width / 2, height - 14 * mm, inst["republique"])
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawCentredString(width / 2, height - 22 * mm, inst["ministere"])
    canvas.drawCentredString(width / 2, height - 28 * mm, inst["hopital"])
    canvas.setStrokeColor(colors.black)
    canvas.setLineWidth(1.2)
    canvas.line(width * 0.33, height - 39 * mm, width * 0.67, height - 39 * mm)
    canvas.setFont("Helvetica-Bold", 16)
    canvas.drawCentredString(width / 2, height - 49 * mm, "FICHE MEDICALE")


def _pdf_footer(canvas, doc, report_data):
    width, _ = A4
    y = 18 * mm
    canvas.setStrokeColor(colors.HexColor("#B9C6D0"))
    canvas.setLineWidth(0.8)
    canvas.line(doc.leftMargin, y + 13 * mm, width - doc.rightMargin, y + 13 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#444444"))
    canvas.drawString(doc.leftMargin, y + 7 * mm, f"Date : {report_data['generated_at'].strftime('%d/%m/%Y')}")
    canvas.drawString(width / 2 - 20 * mm, y + 7 * mm, "Signature")
    canvas.rect(width / 2 - 23 * mm, y - 1 * mm, 40 * mm, 10 * mm, stroke=1, fill=0)
    canvas.drawRightString(width - doc.rightMargin, y + 7 * mm, "Cachet")
    canvas.rect(width - doc.rightMargin - 40 * mm, y - 1 * mm, 40 * mm, 10 * mm, stroke=1, fill=0)
    canvas.setFont("Helvetica", 7)
    canvas.drawRightString(width - doc.rightMargin, 8 * mm, f"Page {canvas.getPageNumber()}")


def build_medical_sheet_pdf(report_data):
    from reportlab.platypus import NextPageTemplate, PageTemplate, BaseDocTemplate, Frame
    from reportlab.lib.pagesizes import A4 as _A4

    buffer = BytesIO()
    PAGE_W, PAGE_H = _A4
    L_MAR = 16 * mm
    R_MAR = 16 * mm
    BOT_MAR = 28 * mm
    TOP_P1 = 62 * mm
    TOP_P2 = 14 * mm
    USABLE_W = PAGE_W - L_MAR - R_MAR  # ~178 mm

    frame_p1 = Frame(L_MAR, BOT_MAR, USABLE_W, PAGE_H - TOP_P1 - BOT_MAR,
                     id="p1", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    frame_p2 = Frame(L_MAR, BOT_MAR, USABLE_W, PAGE_H - TOP_P2 - BOT_MAR,
                     id="p2", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    def _on_page1(canvas, doc):
        _pdf_header(canvas, doc, report_data)
        _pdf_footer(canvas, doc, report_data)

    def _on_page2(canvas, doc):
        _pdf_footer(canvas, doc, report_data)

    pt_p1 = PageTemplate(id="Page1", frames=[frame_p1], onPage=_on_page1)
    pt_p2 = PageTemplate(id="Page2", frames=[frame_p2], onPage=_on_page2)

    doc = BaseDocTemplate(buffer, pagesize=_A4,
                          leftMargin=L_MAR, rightMargin=R_MAR,
                          topMargin=TOP_P1, bottomMargin=BOT_MAR)
    doc.addPageTemplates([pt_p1, pt_p2])

    story = []
    patient = report_data["patient"]
    analysis = report_data["analysis"]

    # Photo patient
    PHOTO_W, PHOTO_H = 36 * mm, 46 * mm
    photo = _rl_image(patient["photo"], PHOTO_W, PHOTO_H) if patient["photo"] else None
    if photo is None:
        photo = Table([["Photo\nnon\nrenseignee"]], colWidths=[PHOTO_W], rowHeights=[PHOTO_H])
        photo.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#B9C6D0")),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))

    # QR code reduit
    qr_payload = (f"{patient['nom']} | {patient['prenom']} | "
                  f"{report_data['generated_at'].strftime('%d/%m/%Y')} | KDIGO {analysis['prediction']}")
    QR_SIZE = 28 * mm
    LEFT_COL_W = 40 * mm
    qr_img = createBarcodeDrawing("QR", value=qr_payload, width=QR_SIZE, height=QR_SIZE, barBorder=0)
    # Pas de Table wrapper pour le QR : on l'insère directement dans left_stack
    # pour éviter l'arriere-plan résiduel de la cellule
    left_stack = Table([[photo], [Spacer(1, 3)], [qr_img]], colWidths=[LEFT_COL_W])
    left_stack.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    GAP = 4 * mm
    RIGHT_COL_W = USABLE_W - LEFT_COL_W - GAP
    ID_COL1 = 52 * mm
    ID_COL2 = RIGHT_COL_W - ID_COL1
    patient_rows = [
        ("N ID", patient["numero_identification"]),
        ("Nom", patient["nom"]),
        ("Prenom", patient["prenom"]),
        ("Telephone", patient["telephone"]),
        ("Adresse", patient["adresse"]),
        ("Date de naissance", patient["date_naissance"]),
        ("Lieu de naissance", patient["lieu_naissance"]),
    ]
    identity_block = Table(
        [[left_stack, _kv_table("IDENTITE DU PATIENT", patient_rows, col_widths=(ID_COL1, ID_COL2))]],
        colWidths=[LEFT_COL_W + GAP, RIGHT_COL_W],
    )
    identity_block.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(identity_block)
    story.append(Spacer(1, 6))

    INFO_COL1 = 60 * mm
    INFO_COL2 = USABLE_W - INFO_COL1

    story.append(_kv_table(
        "INFORMATIONS PATIENT",
        [("Age", patient["age"]), ("Provenance", patient["provenance"]), ("Profession", patient["profession"])],
        col_widths=(INFO_COL1, INFO_COL2),
    ))
    story.append(Spacer(1, 6))
    story.append(_kv_table(
        "RESULTATS",
        [
            ("Classification KDIGO", analysis["prediction"]),
            ("Niveau de risque", analysis["severity"]),
            ("Niveau de confiance", f"{analysis['confidence']:.1f}%"),
            ("Appreciation", analysis["confidence_label"]),
        ],
        col_widths=(INFO_COL1, INFO_COL2),
    ))
    story.append(Spacer(1, 6))

    if analysis["factors"]:
        FACT_COL2 = 30 * mm
        FACT_COL1 = USABLE_W - FACT_COL2
        factor_tbl = Table(
            [("Facteur principal", "Impact")] + analysis["factors"],
            colWidths=[FACT_COL1, FACT_COL2],
            repeatRows=1,
        )
        factor_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#153448")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#C9D9E5")),
            ("BOX", (0, 0), (-1, -1), 0.65, colors.HexColor("#AFC4D4")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FBFD")]),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(NextPageTemplate("Page2"))
        story.append(factor_tbl)
        story.append(Spacer(1, 6))

    story.append(_kv_table(
        "NIVEAU DE GRAVITE",
        [("KDIGO", str(analysis["prediction"])), ("Niveau", analysis["severity"])],
        col_widths=(INFO_COL1, INFO_COL2),
    ))
    story.append(Spacer(1, 6))
    story.append(_bullet_table("PREVENTION", analysis["prevention"], col_width=USABLE_W))
    story.append(Spacer(1, 6))
    story.append(_bullet_table("CONTEXTE TERRAIN", analysis["terrain"], col_width=USABLE_W))

    doc.build(story)
    out = buffer.getvalue()
    buffer.close()
    return out

def analyze_patient(patient_df: pd.DataFrame):
    prediction = int(model.predict(patient_df)[0])
    proba = model.predict_proba(patient_df)[0]
    patient_transformed = model.named_steps["preprocess"].transform(patient_df)
    shap_values_patient = explainer.shap_values(patient_transformed)
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    class_index = list(model.named_steps["classifier"].classes_).index(prediction)
    if isinstance(shap_values_patient, list):
        impacts = shap_values_patient[class_index][0]
    else:
        impacts = shap_values_patient[0, :, class_index]
    shap_df = pd.DataFrame({"feature": feature_names, "impact": impacts}).sort_values(by="impact", ascending=False)
    return prediction, proba, shap_df


def _risk_css_class(risk_label):
    return {
        "Faible": "risk-low",
        "Modéré": "risk-medium",
        "Élevé": "risk-high",
    }.get(str(risk_label), "risk-medium")


def _navigation_target(label):
    mapping = {
        "Tableau de bord": "dashboard",
        "Nouvelle analyse": "new_analysis",
        "Patients": "patients",
        "Analyses": "analyses",
        "Statistiques": "statistics",
        "Paramètres": "settings",
        "Aide / Support": "support",
    }
    return mapping.get(label, "dashboard")


def init_session_state():
    defaults = {
        "authenticated": False,
        "auth_view": "login",
        "current_page": "dashboard",
        "step": 1,
        "patient_meta": {},
        "patient_image": None,
        "analysis_done": False,
        "selected_patient_id": None,
        "user": {
            "name": "Dr Ali Wusi",
            "email": "ali.wusi@nephroalerte.org",
            "role": "Médecin référent",
            "structure": "Hôpital National de Zinder",
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_demo_analytics_data():
    rng = np.random.default_rng(20260511)
    n = 184
    first_names = ["Amina", "Moussa", "Fatou", "Ibrahim", "Hadiza", "Oumarou", "Zara", "Issa", "Nafissa", "Souleymane"]
    last_names = ["Mahamadou", "Garba", "Issoufou", "Adamou", "Abdou", "Saley", "Moussa", "Bello", "Tahirou", "Sani"]
    risks = rng.choice(["Faible", "Modéré", "Élevé"], size=n, p=[0.52, 0.31, 0.17])
    ages = rng.integers(18, 89, size=n)
    sexes = rng.choice(["Femme", "Homme"], size=n, p=[0.54, 0.46])
    analyzed_dates = [datetime.now() - timedelta(days=int(x)) for x in rng.integers(0, 120, size=n)]
    df = pd.DataFrame({
        "patient_id": [f"PAT-{20260000 + i:08d}" for i in range(1, n + 1)],
        "nom": [f"{rng.choice(first_names)} {rng.choice(last_names)}" for _ in range(n)],
        "sexe": sexes,
        "age": ages,
        "provenance": rng.choice(PROVENANCES, size=n, p=[0.46, 0.38, 0.16]),
        "risk_label": risks,
        "risk_score": [round(x, 1) for x in rng.uniform(38, 97, size=n)],
        "status": rng.choice(["Validée", "Suivi planifié", "À recontrôler", "Critique"], size=n, p=[0.55, 0.22, 0.15, 0.08]),
        "analyzed_at": analyzed_dates,
    })
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 30, 45, 60, 75, 100],
        labels=["18-30", "31-45", "46-60", "61-75", "76+"],
        right=True,
    ).astype(str)
    df.loc[df["risk_label"].eq("Élevé") & (df["risk_score"] > 82), "status"] = "Critique"
    return df.sort_values("analyzed_at", ascending=False).reset_index(drop=True)


def get_analytics_data():
    if "analytics_df" not in st.session_state:
        st.session_state.analytics_df = build_demo_analytics_data()
    return st.session_state.analytics_df.copy()


def append_analysis_record(patient_meta, report_data):
    current = get_analytics_data()
    risk_label = report_data["analysis"]["severity"]
    patient_name = f"{patient_meta.get('prenom', '')} {patient_meta.get('nom', '')}".strip() or "Patient non renseigné"
    new_row = pd.DataFrame([{
        "patient_id": patient_meta.get("numero_identification", f"PAT-{datetime.now().strftime('%Y%m%d%H%M%S')}"),
        "nom": patient_name,
        "sexe": patient_meta.get("sexe", "Non renseigné"),
        "age": int(patient_meta.get("age", 0) or 0),
        "provenance": patient_meta.get("provenance", "Non renseigné"),
        "risk_label": risk_label,
        "risk_score": report_data["analysis"]["confidence"],
        "status": "Critique" if risk_label == "Élevé" else "Suivi planifié" if risk_label == "Modéré" else "Validée",
        "analyzed_at": datetime.now(),
    }])
    new_row["age_group"] = pd.cut(
        new_row["age"],
        bins=[17, 30, 45, 60, 75, 100],
        labels=["18-30", "31-45", "46-60", "61-75", "76+"],
        right=True,
    ).astype(str)
    st.session_state.analytics_df = pd.concat([new_row, current], ignore_index=True)


def render_profile_menu():
    user = st.session_state.user
    if hasattr(st, "popover"):
        with st.popover(f"{user['name']}"):
            st.write(user["role"])
            st.caption(user["email"])
            if st.button("Voir le profil", use_container_width=True):
                st.session_state.current_page = "settings"
                st.rerun()
            if st.button("Modifier les informations d’accès", use_container_width=True):
                st.session_state.current_page = "settings"
                st.rerun()
            if st.button("Changer le mot de passe", use_container_width=True):
                st.session_state.current_page = "settings"
                st.rerun()
            if st.button("Se déconnecter", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.current_page = "dashboard"
                st.rerun()
    else:
        with st.expander(user["name"]):
            st.write(user["role"])
            st.caption(user["email"])
            if st.button("Se déconnecter", use_container_width=True):
                st.session_state.authenticated = False
                st.rerun()


def render_top_header():
    logo_col, title_col, profile_col = st.columns([0.65, 5.2, 1.8])
    with logo_col:
        if os.path.exists("Nephroalerte_logo.png"):
            st.image("Nephroalerte_logo.png", width=58)
        else:
            st.markdown("<div class='brand-mark'>N</div>", unsafe_allow_html=True)
    with title_col:
        st.markdown(
            """
            <div class="brand-title">NéphroAlerte</div>
            <div class="brand-subtitle">Intelligence artificielle au service de la santé rénale</div>
            """,
            unsafe_allow_html=True,
        )
    with profile_col:
        st.markdown(
            f"<div class='header-user'>{st.session_state.user['structure']}</div><div class='header-role'>Profil utilisateur</div>",
            unsafe_allow_html=True,
        )
        render_profile_menu()
    st.markdown("<hr style='border:0;border-top:1px solid #DDEAF4;margin:1rem 0 1.15rem 0;'>", unsafe_allow_html=True)


def render_sidebar():
    user = st.session_state.user
    with st.sidebar:
        st.markdown("### ☰ Menu")
        st.caption("Navigation clinique")
        for label in ["Tableau de bord", "Nouvelle analyse", "Patients", "Analyses", "Statistiques", "Paramètres", "Aide / Support"]:
            target = _navigation_target(label)
            if st.button(label, key=f"nav_{target}", use_container_width=True):
                st.session_state.current_page = target
                if target == "new_analysis":
                    st.session_state.step = 1
                    st.session_state.patient_meta = {}
                st.rerun()
        st.markdown(
            f"""
            <div class="sidebar-meta">
                <p><strong>Structure</strong><br>{user['structure']}</p>
                <p><strong>Utilisateur</strong><br>{user['name']}</p>
                <p><strong>Rôle</strong><br>{user['role']}</p>
                <p><strong>Date / heure</strong><br>{datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_login_page():
    st.markdown("<div class='login-wrap'>", unsafe_allow_html=True)
    if os.path.exists("Nephroalerte_logo.png"):
        logo_left, logo_center, logo_right = st.columns([1.5, 1, 1.5])
        with logo_center:
            st.image("Nephroalerte_logo.png", use_container_width=True)
    st.markdown(
        """
        <div class="login-title">NéphroAlerte</div>
        <div class="login-subtitle">Accès sécurisé à la plateforme d’aide à la décision rénale</div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.auth_view == "register":
        with st.form("register_form"):
            st.markdown("<div class='section-title'>Créer un compte</div>", unsafe_allow_html=True)
            full_name = st.text_input("Nom complet")
            structure = st.text_input("Structure de santé")
            email = st.text_input("Email professionnel")
            password = st.text_input("Mot de passe", type="password")
            confirm_password = st.text_input("Confirmer le mot de passe", type="password")
            create_account = st.form_submit_button("Créer le compte", use_container_width=True)
        if create_account:
            if password and password == confirm_password:
                st.session_state.user.update({
                    "name": full_name or "Utilisateur NéphroAlerte",
                    "email": email or "utilisateur@nephroalerte.org",
                    "structure": structure or "Structure de santé",
                })
                st.session_state.authenticated = True
                st.session_state.current_page = "dashboard"
                st.rerun()
            else:
                st.error("Veuillez vérifier le mot de passe et sa confirmation.")
        if st.button("Déjà un compte ? Se connecter", use_container_width=True):
            st.session_state.auth_view = "login"
            st.rerun()
    else:
        with st.form("login_form"):
            st.markdown("<div class='section-title'>Connexion</div>", unsafe_allow_html=True)
            identifier = st.text_input("Email ou identifiant", placeholder="ex. medecin@hopital.ne")
            password = st.text_input("Mot de passe", type="password")
            remember_me = st.checkbox("Se souvenir de moi")
            login_btn = st.form_submit_button("Se connecter", use_container_width=True)
        if login_btn:
            st.session_state.authenticated = True
            st.session_state.user["email"] = identifier or st.session_state.user["email"]
            st.session_state.remember_me = remember_me
            st.session_state.current_page = "dashboard"
            st.rerun()

        col_google, col_microsoft = st.columns(2)
        with col_google:
            if st.button("Se connecter avec Google", use_container_width=True):
                st.session_state.authenticated = True
                st.session_state.current_page = "dashboard"
                st.rerun()
        with col_microsoft:
            if st.button("Se connecter avec Microsoft", use_container_width=True):
                st.session_state.authenticated = True
                st.session_state.current_page = "dashboard"
                st.rerun()
        link_col1, link_col2 = st.columns(2)
        with link_col1:
            if st.button("Mot de passe oublié ?", use_container_width=True):
                st.info("Un flux de réinitialisation pourra être connecté à votre fournisseur d’identité.")
        with link_col2:
            if st.button("Créer un compte", use_container_width=True):
                st.session_state.auth_view = "register"
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def kpi_card(label, value, help_text=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_page():
    df = get_analytics_data()
    total_patients = df["patient_id"].nunique()
    total_analyses = len(df)
    detected_cases = int(df["risk_label"].isin(["Modéré", "Élevé"]).sum())
    high_cases = int(df["risk_label"].eq("Élevé").sum())
    moderate_cases = int(df["risk_label"].eq("Modéré").sum())
    low_cases = int(df["risk_label"].eq("Faible").sum())
    critical_alerts = int((df["risk_label"].eq("Élevé") | df["status"].eq("Critique")).sum())
    detection_rate = detected_cases / max(total_analyses, 1)
    last_analysis = pd.to_datetime(df["analyzed_at"]).max().strftime("%d/%m/%Y %H:%M")

    hero_col, action_col = st.columns([4.4, 1.6])
    with hero_col:
        st.markdown(
            """
            <div class="hero-panel">
                <div class="eyebrow">Tableau de bord institutionnel</div>
                <div class="hero-title">Surveillance précoce du risque rénal</div>
                <div class="hero-copy">Vue consolidée des patients enrôlés, analyses IA, alertes critiques et indicateurs opérationnels pour prioriser le suivi clinique.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with action_col:
        st.write("")
        if st.button("Nouvelle analyse", type="primary", use_container_width=True):
            st.session_state.current_page = "new_analysis"
            st.session_state.step = 1
            st.session_state.patient_meta = {}
            st.rerun()
        st.caption("Accès direct à la saisie patient existante.")

    st.write("")
    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        kpi_card("Patients enrôlés", f"{total_patients:,}".replace(",", " "), "Cohorte active")
    with kpi_cols[1]:
        kpi_card("Analyses réalisées", f"{total_analyses:,}".replace(",", " "), "Toutes périodes")
    with kpi_cols[2]:
        kpi_card("Cas détectés", f"{detected_cases}", f"{detection_rate:.1%} de détection")
    with kpi_cols[3]:
        kpi_card("Risque élevé", f"{high_cases}", f"{critical_alerts} alertes critiques")
    with kpi_cols[4]:
        kpi_card("Dernière analyse", last_analysis, "Mise à jour automatique")

    kpi_cols_2 = st.columns(4)
    with kpi_cols_2[0]:
        kpi_card("Risque faible", f"{low_cases}", "Prévention standard")
    with kpi_cols_2[1]:
        kpi_card("Risque modéré", f"{moderate_cases}", "Suivi rapproché")
    with kpi_cols_2[2]:
        kpi_card("Taux de détection", f"{detection_rate:.1%}", "Modéré + élevé")
    with kpi_cols_2[3]:
        kpi_card("Alertes critiques", f"{critical_alerts}", "Priorité médicale")

    st.write("")
    chart_col1, chart_col2 = st.columns(2)
    risk_order = ["Faible", "Modéré", "Élevé"]
    risk_counts = df.groupby("risk_label").size().reindex(risk_order, fill_value=0).reset_index(name="patients")
    age_counts = df.groupby("age_group").size().reset_index(name="patients")
    sex_counts = df.groupby("sexe").size().reset_index(name="patients")
    trend = df.assign(day=pd.to_datetime(df["analyzed_at"]).dt.date).groupby("day").size().reset_index(name="analyses").sort_values("day")

    with chart_col1:
        st.markdown("<div class='section-title'>Répartition par niveau de risque</div>", unsafe_allow_html=True)
        if alt is not None:
            risk_chart = alt.Chart(risk_counts).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("risk_label:N", title="Niveau de risque", sort=risk_order),
                y=alt.Y("patients:Q", title="Patients"),
                color=alt.Color("risk_label:N", scale=alt.Scale(domain=risk_order, range=["#2E8B57", "#F2B84B", "#D64545"]), legend=None),
                tooltip=["risk_label", "patients"],
            ).properties(height=270)
            st.altair_chart(risk_chart, use_container_width=True)
        else:
            st.bar_chart(risk_counts.set_index("risk_label"))
    with chart_col2:
        st.markdown("<div class='section-title'>Tendance des analyses dans le temps</div>", unsafe_allow_html=True)
        if alt is not None:
            trend_chart = alt.Chart(trend).mark_line(point=True, strokeWidth=3, color="#1E88E5").encode(
                x=alt.X("day:T", title="Date"),
                y=alt.Y("analyses:Q", title="Analyses"),
                tooltip=["day:T", "analyses"],
            ).properties(height=270)
            st.altair_chart(trend_chart, use_container_width=True)
        else:
            st.line_chart(trend.set_index("day"))

    chart_col3, chart_col4, chart_col5 = st.columns([1.3, 1.1, 1])
    with chart_col3:
        st.markdown("<div class='section-title'>Tranches d’âge</div>", unsafe_allow_html=True)
        if alt is not None:
            age_chart = alt.Chart(age_counts).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#0D5AA7").encode(
                x=alt.X("age_group:N", title="Âge", sort=["18-30", "31-45", "46-60", "61-75", "76+"]),
                y=alt.Y("patients:Q", title="Patients"),
                tooltip=["age_group", "patients"],
            ).properties(height=245)
            st.altair_chart(age_chart, use_container_width=True)
        else:
            st.bar_chart(age_counts.set_index("age_group"))
    with chart_col4:
        st.markdown("<div class='section-title'>Répartition par sexe</div>", unsafe_allow_html=True)
        if alt is not None:
            sex_chart = alt.Chart(sex_counts).mark_arc(innerRadius=58, outerRadius=95).encode(
                theta=alt.Theta("patients:Q"),
                color=alt.Color("sexe:N", scale=alt.Scale(range=["#1E88E5", "#2E8B57", "#94A3B8"])),
                tooltip=["sexe", "patients"],
            ).properties(height=245)
            st.altair_chart(sex_chart, use_container_width=True)
        else:
            st.bar_chart(sex_counts.set_index("sexe"))
    with chart_col5:
        st.markdown("<div class='section-title'>Zone géographique à risque</div>", unsafe_allow_html=True)
        location_risk = df[df["risk_label"].eq("Élevé")].groupby("provenance").size().sort_values(ascending=False)
        top_location = location_risk.index[0] if not location_risk.empty else "Non renseignée"
        top_count = int(location_risk.iloc[0]) if not location_risk.empty else 0
        st.markdown(
            f"""
            <div class="content-panel">
                <div class="kpi-label">Concentration actuelle</div>
                <div class="kpi-value">{top_location}</div>
                <div class="kpi-help">{top_count} patients à risque élevé dans cette zone. À confirmer avec les données terrain.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    latest_col, alert_col = st.columns([1.45, 1])
    with latest_col:
        st.markdown("<div class='section-title'>Dernières analyses</div>", unsafe_allow_html=True)
        header_cols = st.columns([1.05, 1.65, 1.1, 1, 0.8])
        for col, label in zip(header_cols, ["Identifiant", "Patient", "Date", "Risque", ""]):
            col.caption(label)
        latest = df.head(7)
        for _, row in latest.iterrows():
            cols = st.columns([1.05, 1.65, 1.1, 1, 0.8])
            cols[0].write(row["patient_id"])
            cols[1].write(row["nom"])
            cols[2].write(pd.to_datetime(row["analyzed_at"]).strftime("%d/%m/%Y"))
            cols[3].markdown(f"<span class='risk-pill {_risk_css_class(row['risk_label'])}'>{row['risk_label']}</span>", unsafe_allow_html=True)
            if cols[4].button("Détails", key=f"detail_{row['patient_id']}"):
                st.session_state.selected_patient_id = row["patient_id"]
        if st.session_state.selected_patient_id:
            selected = df[df["patient_id"].eq(st.session_state.selected_patient_id)].head(1)
            if not selected.empty:
                patient = selected.iloc[0]
                st.info(
                    f"{patient['patient_id']} - {patient['nom']} - {patient['age']} ans - "
                    f"risque {patient['risk_label']} - statut {patient['status']}."
                )
    with alert_col:
        st.markdown("<div class='section-title'>Alertes et suivi</div>", unsafe_allow_html=True)
        watchlist = df[(df["risk_label"].isin(["Modéré", "Élevé"])) | (df["status"].isin(["Critique", "À recontrôler"]))].head(6)
        for _, row in watchlist.iterrows():
            st.markdown(
                f"""
                <div class="content-panel" style="padding:0.85rem;margin-bottom:0.65rem;">
                    <strong>{row['patient_id']}</strong> - {row['nom']}<br>
                    <span class="risk-pill {_risk_css_class(row['risk_label'])}">{row['risk_label']}</span>
                    <span style="color:#111827;font-weight:700;">{row['status']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_placeholder_page(title, subtitle):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="content-panel">
            <p>{subtitle}</p>
            <p class="empty-note">Cette section est prête pour la connexion à la base de données et aux workflows métier.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_new_analysis_page():
    st.markdown("<div class='section-title'>Nouvelle analyse patient</div>", unsafe_allow_html=True)
    st.caption("Saisie clinique, classification KDIGO, explication IA et génération de fiche médicale.")
    top_col1, top_col2 = st.columns([1, 4])
    with top_col1:
        if st.button("Retour au tableau de bord", use_container_width=True):
            st.session_state.current_page = "dashboard"
            st.rerun()

    if "step" not in st.session_state:
        st.session_state.step = 1
    if "patient_meta" not in st.session_state:
        st.session_state.patient_meta = {}
    if "patient_image" not in st.session_state:
        st.session_state.patient_image = None
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False

    if st.session_state.step == 1:
        st.markdown("<div class='step-heading'><span class='step-index'>01</span><div><div class='section-title'>Informations personnelles</div><div class='step-copy'>Identité et coordonnées du patient</div></div></div>", unsafe_allow_html=True)
        with st.form("step1_form"):
            identity_cols = st.columns([1, 1, 1])
            numero_identification = identity_cols[0].text_input("Numéro d’identification", value=f"PAT-{datetime.now().strftime('%Y%m%d%H%M%S')}", disabled=True)
            nom = identity_cols[1].text_input("Nom")
            prenom = identity_cols[2].text_input("Prénom")
            contact_cols = st.columns([1, 1, 1])
            sexe = contact_cols[0].selectbox("Sexe", ["Femme", "Homme", "Non renseigné"])
            telephone = contact_cols[1].text_input("Téléphone")
            date_naissance = contact_cols[2].date_input("Date de naissance", value=date(1985, 1, 1))
            location_cols = st.columns([1, 2])
            lieu_naissance = location_cols[0].text_input("Lieu de naissance")
            adresse = location_cols[1].text_input("Adresse")
            photo_patient = st.file_uploader("Photo du patient", type=["png", "jpg", "jpeg"])
            next_btn = st.form_submit_button("Continuer →", type="primary", use_container_width=True)
        if next_btn:
            st.session_state.patient_meta.update({
                "numero_identification": numero_identification,
                "nom": nom,
                "prenom": prenom,
                "sexe": sexe,
                "telephone": telephone,
                "adresse": adresse,
                "date_naissance": date_naissance.strftime("%d/%m/%Y") if hasattr(date_naissance, "strftime") else str(date_naissance),
                "lieu_naissance": lieu_naissance,
                "photo_patient": photo_patient,
            })
            st.session_state.step = 2
            st.rerun()

    elif st.session_state.step == 2:
        st.markdown("<div class='step-heading'><span class='step-index'>02</span><div><div class='section-title'>Informations cliniques</div><div class='step-copy'>Complétez les paramètres généraux, les constantes et les examens biologiques</div></div></div>", unsafe_allow_html=True)
        with st.form("step2_form"):
            st.markdown("<div class='form-section-title'>Informations générales</div>", unsafe_allow_html=True)
            general_cols = st.columns(3)
            age = general_cols[0].selectbox("Âge", list(range(18, 91)))
            provenance = general_cols[1].selectbox("Provenance", PROVENANCES)
            profession = general_cols[2].selectbox("Profession", PROFESSIONS)
            history_cols = st.columns(2)
            antecedents = history_cols[0].selectbox("Antécédents", ANTECEDENTS)
            hydratation = history_cols[1].selectbox("Hydratation", HYDRATATIONS)
            st.markdown("<div class='form-section-title'>Constantes physiologiques</div>", unsafe_allow_html=True)
            vital_cols = st.columns(3)
            pa_systolique = vital_cols[0].number_input("PA systolique", min_value=50, max_value=250, value=120)
            pa_diastolique = vital_cols[1].number_input("PA diastolique", min_value=30, max_value=180, value=80)
            diurese_ml_24h = vital_cols[2].number_input("Diurèse ml / 24h", min_value=0, max_value=5000, value=1200)
            st.markdown("<div class='form-section-title'>Examens biologiques</div>", unsafe_allow_html=True)
            lab_cols = st.columns(3)
            creatininemie_mg_dl = lab_cols[0].number_input("Créatininémie (mg/dL)", min_value=0.1, max_value=20.0, value=1.0, step=0.1)
            azotemie_mg_dl = lab_cols[1].number_input("Azotémie (mg/dL)", min_value=0.0, max_value=300.0, value=35.0, step=0.1)
            hb_g_dl = lab_cols[2].number_input("Hb (g/dL)", min_value=0.0, max_value=25.0, value=13.0, step=0.1)
            blood_cols = st.columns(2)
            gb_10e3_ul = blood_cols[0].number_input("GB (10e3/µL)", min_value=0.0, max_value=100.0, value=8.0, step=0.1)
            plaquettes_10e3_ul = blood_cols[1].number_input("Plaquettes (10e3/µL)", min_value=0, max_value=1000, value=250)
            col1, col2 = st.columns(2)
            with col1:
                back_btn = st.form_submit_button("← Retour", use_container_width=True)
            with col2:
                analyze_btn = st.form_submit_button("Analyser", type="primary", use_container_width=True)
        if back_btn:
            st.session_state.step = 1
            st.rerun()
        if analyze_btn:
            st.session_state.patient_meta.update({
                "age": age,
                "provenance": provenance,
                "profession": profession,
                "antecedents": antecedents,
                "hydratation": hydratation,
                "pa_systolique": pa_systolique,
                "pa_diastolique": pa_diastolique,
                "diurese_ml_24h": diurese_ml_24h,
                "creatininemie_mg_dl": creatininemie_mg_dl,
                "azotemie_mg_dl": azotemie_mg_dl,
                "hb_g_dl": hb_g_dl,
                "gb_10e3_ul": gb_10e3_ul,
                "plaquettes_10e3_ul": plaquettes_10e3_ul,
            })
            patient_df = pd.DataFrame([{
                "age": age,
                "provenance": provenance,
                "profession": profession,
                "antecedents": antecedents,
                "pa_systolique": pa_systolique,
                "pa_diastolique": pa_diastolique,
                "hydratation": hydratation,
                "diurese_ml_24h": diurese_ml_24h,
                "creatininemie_mg_dl": creatininemie_mg_dl,
                "azotemie_mg_dl": azotemie_mg_dl,
                "hb_g_dl": hb_g_dl,
                "gb_10e3_ul": gb_10e3_ul,
                "plaquettes_10e3_ul": plaquettes_10e3_ul,
            }])
            progress = st.progress(0, text="Préparation des données...")
            time.sleep(0.2)
            progress.progress(30, text="Prétraitement en cours...")
            time.sleep(0.2)
            progress.progress(60, text="Analyse du modèle IA...")
            time.sleep(0.2)
            prediction, proba, shap_df = analyze_patient(patient_df)
            progress.progress(100, text="Analyse terminée")
            report_data = build_report_data(st.session_state.patient_meta, prediction, proba, shap_df)
            append_analysis_record(st.session_state.patient_meta, report_data)
            st.success("Analyse terminée avec succès.")
            st.markdown("<div class='section-title result-heading'>Résultat de l’analyse</div>", unsafe_allow_html=True)
            metric_col1, metric_col2, metric_col3 = st.columns([1.2, 1, 1])
            metric_col1.metric("Classe KDIGO", str(prediction))
            metric_col2.metric("Niveau de risque", report_data["analysis"]["severity"])
            metric_col3.metric("Confiance du modèle", f"{report_data['analysis']['confidence']:.1f}%")
            tab1, tab2, tab3 = st.tabs(["Fiche médicale", "Facteurs influents", "Données brutes"])
            with tab1:
                render_medical_sheet(report_data)
                pdf_bytes = build_medical_sheet_pdf(report_data)
                st.download_button("Télécharger le rapport PDF", pdf_bytes, file_name="fiche_medicale_kdigo.pdf", mime="application/pdf")
            with tab2:
                top_features = shap_df.head(10).copy()
                top_features["feature"] = top_features["feature"].map(lambda x: FEATURE_MAPPING.get(x, x))
                st.dataframe(top_features, use_container_width=True)
            with tab3:
                st.dataframe(patient_df, use_container_width=True)
                st.dataframe(pd.DataFrame([proba], columns=[f"classe_{c}" for c in model.named_steps["classifier"].classes_]), use_container_width=True)


def main():
    init_session_state()
    if not st.session_state.authenticated:
        render_login_page()
        return

    render_sidebar()
    render_top_header()

    page = st.session_state.current_page
    if page == "dashboard":
        render_dashboard_page()
    elif page == "new_analysis":
        render_new_analysis_page()
    elif page == "patients":
        render_placeholder_page("Patients", "Registre des patients enrôlés, recherche, filtres et accès aux fiches individuelles.")
        st.dataframe(get_analytics_data()[["patient_id", "nom", "sexe", "age", "provenance", "risk_label", "status"]], use_container_width=True)
    elif page == "analyses":
        render_placeholder_page("Analyses", "Historique des analyses réalisées, validations médicales et exports.")
        st.dataframe(get_analytics_data()[["patient_id", "nom", "analyzed_at", "risk_label", "risk_score", "status"]], use_container_width=True)
    elif page == "statistics":
        render_dashboard_page()
    elif page == "settings":
        render_placeholder_page("Paramètres", "Gestion du profil, des accès, des rôles et des préférences institutionnelles.")
    elif page == "support":
        render_placeholder_page("Aide / Support", "Documentation, support utilisateur et procédures de contact.")
    else:
        render_dashboard_page()


main()

