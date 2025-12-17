"""
🏥 PLATEFORME MÉDICALE ULTIME - XAI Multi-Techniques Professionnel
✅ Authentification Flask-Login
✅ LIME pour métadonnées
✅ Métriques de confiance clinique
✅ Interface médicale complète
✅ VRAI MODÈLE UNIQUEMENT (pas de simulation)
✅ 📄 Génération de rapports PDF médicaux
✅ 🎨 Visualisations 3D interactives (CORRIGÉ)
"""

import os
import io
import base64
import sys
import numpy as np
import cv2
from flask import Flask, render_template, jsonify, request, send_file, redirect, url_for
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import json

# Nouvelles dépendances
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Authentification
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# =======================================================
# CONFIGURATION & AUTHENTIFICATION
# =======================================================

# Initialiser les extensions AVANT l'app
db = SQLAlchemy()
login_manager = LoginManager()


# Modèle utilisateur
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    full_name = db.Column(db.String(100))
    institution = db.Column(db.String(200))
    role = db.Column(db.String(50), default='medecin')
    created_at = db.Column(db.DateTime, default=datetime.now)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Créer l'application Flask
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'medical-xai-secret-key-2025-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medical_xai.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialiser extensions
CORS(app)
db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = 'login'

BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
RESULTS_FOLDER = BASE_DIR / 'static' / 'results'
REPORTS_FOLDER = BASE_DIR / 'reports'
UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)
RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)
REPORTS_FOLDER.mkdir(exist_ok=True, parents=True)

CLASS_NAMES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC', 'SCC']
CLASS_COLORS = {
    'AKIEC': '#FF6B6B', 'BCC': '#4ECDC4', 'BKL': '#45B7D1', 'DF': '#96CEB4',
    'MEL': '#FFEAA7', 'NV': '#DDA0DD', 'VASC': '#FFA07A', 'SCC': '#98D8C8'
}
NV_INDEX = 5
VASC_INDEX = 6


# =======================================================
# CLASSE GENERATEUR DE RAPPORTS PDF
# =======================================================

class MedicalPDFGenerator:
    """Génère des rapports PDF médicaux professionnels"""

    def __init__(self, reports_folder=REPORTS_FOLDER):
        self.reports_folder = Path(reports_folder)
        self.reports_folder.mkdir(exist_ok=True)

    def create_pdf_report(self, analysis_data, patient_info=None, doctor_info=None):
        """Crée un rapport PDF complet"""

        # Créer un nom de fichier unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"medical_report_{analysis_data.get('analysis_id', timestamp)}_{timestamp}.pdf"
        filepath = self.reports_folder / filename

        # Créer le document
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        # Styles
        styles = getSampleStyleSheet()

        # Styles personnalisés
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30
        )

        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=12
        )

        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.black,
            spaceAfter=6
        )

        # Contenu du PDF
        story = []

        # 1. EN-TÊTE
        story.append(Paragraph("<b>RAPPORT MEDICAL D'ANALYSE PAR IA</b>", title_style))
        story.append(Spacer(1, 12))

        # Informations patient/médecin
        patient_data = patient_info or {}
        doctor_data = doctor_info or {
            'name': analysis_data.get('user_info', {}).get('full_name', 'Dr. Non Spécifié'),
            'institution': analysis_data.get('user_info', {}).get('institution', 'Centre Médical')
        }

        info_table_data = [
            ['<b>INFORMATIONS PATIENT</b>', '<b>MEDECIN RESPONSABLE</b>', '<b>ANALYSE</b>'],
            [
                f"Nom: {patient_data.get('name', 'Non spécifié')}<br/>"
                f"Age: {patient_data.get('age', 'N/A')}<br/>"
                f"Sexe: {patient_data.get('sex', 'N/A')}",

                f"Dr. {doctor_data.get('name', '')}<br/>"
                f"{doctor_data.get('institution', '')}<br/>"
                f"N° RPPS: {doctor_data.get('rpps', 'N/A')}",

                f"Ref: {analysis_data.get('analysis_id', 'N/A')}<br/>"
                f"Date: {analysis_data.get('timestamp', datetime.now().strftime('%d/%m/%Y'))}<br/>"
                f"Heure: {datetime.now().strftime('%H:%M')}"
            ]
        ]

        info_table = Table(info_table_data, colWidths=[6 * cm, 6 * cm, 4 * cm])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#ecf0f1')),
            ('FONTSIZE', (0, 1), (-1, 1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))

        story.append(info_table)
        story.append(Spacer(1, 20))

        # 2. DIAGNOSTIC PRINCIPAL
        story.append(Paragraph("<b>DIAGNOSTIC PRINCIPAL</b>", subtitle_style))

        predicted_class = analysis_data.get('predicted_class', 'N/A')
        confidence = analysis_data.get('confidence', 0) * 100

        # Couleur selon la confiance
        if confidence >= 75:
            confidence_color = colors.HexColor('#27ae60')  # Vert
        elif confidence >= 50:
            confidence_color = colors.HexColor('#f39c12')  # Orange
        else:
            confidence_color = colors.HexColor('#e74c3c')  # Rouge

        diagnostic_text = f"""
        <b>Pathologie détectée :</b> {predicted_class}<br/>
        <b>Confiance du modèle :</b> <font color="{confidence_color.hexval()}">{confidence:.1f}%</font><br/>
        <b>Niveau de confiance XAI :</b> {analysis_data.get('confidence_metrics', {}).get('confidence_level', 'N/A')}
        """

        story.append(Paragraph(diagnostic_text, normal_style))
        story.append(Spacer(1, 10))

        # 3. DISTRIBUTION DES PROBABILITES
        story.append(Paragraph("<b>DISTRIBUTION DES PROBABILITES</b>", subtitle_style))

        probs = analysis_data.get('probabilities', {})
        prob_data = [['PATHOLOGIE', 'PROBABILITE', 'NIVEAU']]

        for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            prob_percent = prob * 100
            if prob_percent > 75:
                level = "ELEVE"
                color = colors.green
            elif prob_percent > 50:
                level = "MODERE"
                color = colors.orange
            elif prob_percent > 25:
                level = "FAIBLE"
                color = colors.yellow
            else:
                level = "TRES FAIBLE"
                color = colors.lightgrey

            prob_data.append([cls, f"{prob_percent:.1f}%", level])

        prob_table = Table(prob_data, colWidths=[5 * cm, 3 * cm, 4 * cm])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f8f9fa')),
            ('TEXTCOLOR', (2, 1), (2, -1), colors.HexColor('#2c3e50')),
        ]))

        story.append(prob_table)
        story.append(Spacer(1, 20))

        # 4. METRIQUES D'EXPLICABILITE
        story.append(Paragraph("<b>METRIQUES D'EXPLICABILITE (XAI)</b>", subtitle_style))

        xai_metrics = analysis_data.get('xai_metrics', {})
        conf_metrics = analysis_data.get('confidence_metrics', {})

        xai_data = [
            ['METRIQUE', 'VALEUR', 'INTERPRETATION'],
            ['Force du consensus', f"{conf_metrics.get('consensus_strength', 0):.2f}",
             "Accord entre les méthodes XAI"],
            ['Accord moyen', f"{conf_metrics.get('avg_agreement', 0):.2f}",
             "Corrélation entre techniques"],
            ['Sparsité', f"{conf_metrics.get('sparsity', 0):.2f}",
             "Focalisation de l'explication"],
            ['Score composite', f"{conf_metrics.get('confidence_score', 0):.2f}",
             conf_metrics.get('confidence_level', 'N/A')],
        ]

        xai_table = Table(xai_data, colWidths=[4 * cm, 3 * cm, 7 * cm])
        xai_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b59b6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#f8f9fa')),
            ('ALIGN', (2, 1), (2, -1), 'LEFT'),
        ]))

        story.append(xai_table)
        story.append(Spacer(1, 15))

        # Corrélations
        correlations = conf_metrics.get('correlations', {})
        if correlations:
            corr_text = "<b>Corrélations entre méthodes :</b><br/>"
            for method, value in correlations.items():
                corr_text += f"• {method}: {value:.2f}<br/>"
            story.append(Paragraph(corr_text, normal_style))

        # 5. ANALYSE LIME DES METADONNEES
        if analysis_data.get('lime_interpretation'):
            story.append(Paragraph("<b>ANALYSE DES METADONNEES (LIME)</b>", subtitle_style))
            story.append(Paragraph(analysis_data['lime_interpretation'].replace('\n', '<br/>'), normal_style))
            story.append(Spacer(1, 10))

        # 6. METHODES UTILISEES
        story.append(Paragraph("<b>METHODES D'EXPLICATION UTILISEES</b>", subtitle_style))

        methods = analysis_data.get('xai_methods', [])
        methods_text = "• " + "<br/>• ".join(methods)
        story.append(Paragraph(methods_text, normal_style))
        story.append(Spacer(1, 15))

        # 7. AVERTISSEMENT MEDICAL
        story.append(Paragraph("<b>AVERTISSEMENT MEDICAL</b>", subtitle_style))

        warning_text = """
        Ce rapport est généré par un système d'intelligence artificielle et doit être considéré 
        comme une aide au diagnostic. Il ne remplace pas l'expertise clinique d'un professionnel 
        de santé. Le médecin responsable doit valider les résultats et prendre la décision 
        thérapeutique finale.

        Les résultats présentés sont basés sur des modèles statistiques et peuvent contenir 
        des erreurs. En cas de doute, consulter un spécialiste ou effectuer des examens 
        complémentaires.
        """

        warning_style = ParagraphStyle(
            'WarningStyle',
            parent=normal_style,
            textColor=colors.red,
            fontSize=9,
            backColor=colors.HexColor('#fff3cd'),
            borderWidth=1,
            borderColor=colors.HexColor('#ffeaa7'),
            borderPadding=10,
            spaceAfter=20
        )

        story.append(Paragraph(warning_text, warning_style))

        # 8. SIGNATURE
        story.append(Spacer(1, 30))
        signature_text = f"""
        Rapport généré automatiquement le {datetime.now().strftime('%d/%m/%Y à %H:%M')}<br/>
        Plateforme Médicale XAI - Système d'Aide au Diagnostic<br/>
        <i>Document à usage médical uniquement</i>
        """
        story.append(Paragraph(signature_text, ParagraphStyle(
            'Signature',
            parent=normal_style,
            alignment=TA_CENTER,
            fontSize=8,
            textColor=colors.grey
        )))

        # Générer le PDF
        doc.build(story)

        return {
            'filename': filename,
            'filepath': str(filepath),
            'url': f'/api/download_pdf/{analysis_data.get("analysis_id", timestamp)}'
        }


# =======================================================
# CLASSE VISUALISATEUR 3D (AMÉLIORÉE)
# =======================================================

class XAI3DVisualizer:
    """Génère des visualisations 3D interactives des cartes XAI"""

    def __init__(self):
        pass

    def create_3d_surface_json(self, heatmap, image=None, title="Carte XAI 3D", colorscale='Viridis'):
        """Crée une surface 3D d'une carte de chaleur - retourne JSON pour Plotly"""

        x = np.arange(heatmap.shape[1])
        y = np.arange(heatmap.shape[0])
        X, Y = np.meshgrid(x, y)
        Z = heatmap

        fig = go.Figure(data=[
            go.Surface(
                z=Z,
                x=X,
                y=Y,
                colorscale=colorscale,
                opacity=0.9,
                contours={
                    "z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project": {"z": True}}
                },
                name='Carte XAI'
            )
        ])

        if image is not None:
            # Ajouter l'image comme base
            fig.add_trace(go.Surface(
                z=np.zeros_like(Z) - 0.1,
                x=X,
                y=Y,
                surfacecolor=image if len(image.shape) == 3 else np.stack([image] * 3, axis=-1),
                colorscale=[(0, 'white'), (1, 'white')],
                opacity=0.7,
                showscale=False,
                name='Image originale'
            ))

        fig.update_layout(
            title=title,
            scene={
                'xaxis': {'title': 'Largeur (px)', 'backgroundcolor': 'white'},
                'yaxis': {'title': 'Hauteur (px)', 'backgroundcolor': 'white'},
                'zaxis': {'title': 'Importance', 'backgroundcolor': 'white'},
                'camera': {'eye': {'x': 1.5, 'y': 1.5, 'z': 1.2}},
                'aspectmode': 'manual',
                'aspectratio': {'x': 1, 'y': 1, 'z': 0.5}
            },
            margin={'l': 0, 'r': 0, 'b': 0, 't': 40},
            height=600,
            showlegend=True
        )

        # Retourner le JSON de la figure
        return fig.to_dict()

    def create_multi_3d_view_json(self, occlusion, intgrad, attention, consensus, image=None):
        """Crée une vue 3D multi-techniques (4-en-1) - retourne JSON"""

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Occlusion Sensitivity', 'Integrated Gradients',
                            'Attention Rollout', 'Consensus Map'),
            specs=[[{'type': 'surface'}, {'type': 'surface'}],
                   [{'type': 'surface'}, {'type': 'surface'}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )

        maps = [occlusion, intgrad, attention, consensus]
        titles = ['Occlusion', 'IntGrad', 'Attention', 'Consensus']
        colorscales = ['Hot', 'Electric', 'Blues', 'Viridis']

        for idx, (heatmap, title, colorscale) in enumerate(zip(maps, titles, colorscales)):
            row = idx // 2 + 1
            col = idx % 2 + 1

            x = np.arange(heatmap.shape[1])
            y = np.arange(heatmap.shape[0])
            X, Y = np.meshgrid(x, y)
            Z = heatmap

            fig.add_trace(
                go.Surface(
                    z=Z, x=X, y=Y,
                    colorscale=colorscale,
                    opacity=0.85,
                    name=title,
                    showscale=True
                ),
                row=row, col=col
            )

        fig.update_layout(
            title='Vue Multi-Techniques 3D - Cartes XAI',
            height=800,
            margin={'l': 0, 'r': 0, 'b': 0, 't': 60},
            showlegend=True
        )

        return fig.to_dict()

    def create_3d_surface_html(self, heatmap, image=None, title="Carte XAI 3D", colorscale='Viridis'):
        """Crée une surface 3D - version HTML pour compatibilité"""
        fig_dict = self.create_3d_surface_json(heatmap, image, title, colorscale)
        return pio.to_html(fig_dict, full_html=False, include_plotlyjs='cdn')

    def create_multi_3d_view_html(self, occlusion, intgrad, attention, consensus, image=None):
        """Crée une vue 3D multi-techniques - version HTML pour compatibilité"""
        fig_dict = self.create_multi_3d_view_json(occlusion, intgrad, attention, consensus, image)
        return pio.to_html(fig_dict, full_html=False, include_plotlyjs='cdn')


# =======================================================
# CLASSE XAI MULTI-TECHNIQUES
# =======================================================

class UltimateXAIEngine:
    """Moteur XAI avec 3 techniques + LIME"""

    def __init__(self, model, img_size, is_multi_input, num_metadata=0):
        self.model = model
        self.img_size = img_size
        self.is_multi_input = is_multi_input
        self.num_metadata = num_metadata
        print("✅ XAI Engine initialisé")

    def occlusion_sensitivity(self, image, metadata, target_class, patch_size=48, stride=24):
        print(f"   🔲 Occlusion (patch={patch_size}, stride={stride})...")
        h, w = image.shape[:2]
        img_batch = np.expand_dims(image, axis=0)
        meta_batch = np.expand_dims(metadata, axis=0)
        inputs = [img_batch, meta_batch] if self.is_multi_input else img_batch

        base_pred = self.model.predict(inputs, verbose=0)[0]
        base_score = base_pred[target_class]
        importance_map = np.zeros((h, w))
        n_patches = 0

        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                img_occluded = image.copy()
                img_occluded[y:y + patch_size, x:x + patch_size] = 0.5
                img_occ_batch = np.expand_dims(img_occluded, axis=0)
                inputs_occ = [img_occ_batch, meta_batch] if self.is_multi_input else img_occ_batch
                occ_pred = self.model.predict(inputs_occ, verbose=0)[0]
                occ_score = occ_pred[target_class]
                importance = base_score - occ_score
                importance_map[y:y + patch_size, x:x + patch_size] = max(importance, 0)
                n_patches += 1

        if importance_map.max() > 0:
            importance_map = importance_map / importance_map.max()
        print(f"      ✓ {n_patches} patches testés")
        return importance_map

    def integrated_gradients(self, image, metadata, target_class, steps=30):
        print(f"   📊 Integrated Gradients (steps={steps})...")
        baseline = np.zeros_like(image)
        alphas = np.linspace(0, 1, steps)
        integrated_grads = np.zeros_like(image)

        for alpha in alphas:
            img_interp = baseline + alpha * (image - baseline)
            img_batch = np.expand_dims(img_interp, axis=0)
            meta_batch = np.expand_dims(metadata, axis=0)
            img_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)
            meta_tensor = tf.convert_to_tensor(meta_batch, dtype=tf.float32)

            with tf.GradientTape() as tape:
                tape.watch(img_tensor)
                inputs = [img_tensor, meta_tensor] if self.is_multi_input else img_tensor
                predictions = self.model(inputs, training=False)
                target_output = predictions[0, target_class]
            grads = tape.gradient(target_output, img_tensor)
            if grads is not None:
                integrated_grads += grads[0].numpy()

        integrated_grads = integrated_grads / steps
        integrated_grads = integrated_grads * (image - baseline)
        attribution_map = np.sum(np.abs(integrated_grads), axis=-1)
        if attribution_map.max() > 0:
            attribution_map = attribution_map / attribution_map.max()
        print(f"      ✓ Gradients intégrés calculés")
        return attribution_map

    def attention_rollout(self, image, metadata, blur_sigma=10):
        print(f"   🎯 Attention Rollout (blur={blur_sigma})...")
        img_batch = np.expand_dims(image, axis=0)
        meta_batch = np.expand_dims(metadata, axis=0)
        img_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)
        meta_tensor = tf.convert_to_tensor(meta_batch, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            inputs = [img_tensor, meta_tensor] if self.is_multi_input else img_tensor
            predictions = self.model(inputs, training=False)
            output_sum = tf.reduce_sum(predictions)
        grads = tape.gradient(output_sum, img_tensor)

        if grads is not None:
            grads_np = grads[0].numpy()
            attention = np.sum(np.abs(grads_np), axis=-1)
            attention = gaussian_filter(attention, sigma=blur_sigma)
            if attention.max() > 0:
                attention = attention / attention.max()
        else:
            attention = np.zeros(image.shape[:2])
        print(f"      ✓ Carte d'attention générée")
        return attention

    def compute_consensus(self, occlusion, intgrad, attention):
        return (occlusion + intgrad + attention) / 3

    def compute_correlations(self, occlusion, intgrad, attention):
        occ_flat = occlusion.flatten()
        ig_flat = intgrad.flatten()
        att_flat = attention.flatten()
        return {
            'occ_ig': float(np.corrcoef(occ_flat, ig_flat)[0, 1]),
            'occ_att': float(np.corrcoef(occ_flat, att_flat)[0, 1]),
            'ig_att': float(np.corrcoef(ig_flat, att_flat)[0, 1])
        }

    def compute_confidence_metrics(self, occlusion, intgrad, attention):
        """Calcule les métriques de confiance clinique"""
        total_pixels = occlusion.size

        # 1. Sparsité (plus c'est concentré, mieux c'est)
        active_pixels_occ = np.sum(occlusion > 0.3)
        sparsity = 1.0 - (active_pixels_occ / total_pixels)

        # 2. Consensus et accord
        consensus = self.compute_consensus(occlusion, intgrad, attention)
        consensus_strength = consensus.max()
        correlations = self.compute_correlations(occlusion, intgrad, attention)
        avg_agreement = np.mean(list(correlations.values()))

        # 3. Score de confiance composite (0-1)
        confidence_score = (
                0.4 * consensus_strength +
                0.3 * avg_agreement +
                0.2 * sparsity +
                0.1 * (occlusion.max() + intgrad.max() + attention.max()) / 3
        )

        # 4. Niveau de confiance
        if confidence_score > 0.75:
            confidence_level = "ELEVE"
            confidence_color = "success"
            recommendation = "Prise en charge selon protocole"
        elif confidence_score > 0.5:
            confidence_level = "MODERE"
            confidence_color = "warning"
            recommendation = "Validation par dermatologue recommandée"
        else:
            confidence_level = "LIMITE"
            confidence_color = "danger"
            recommendation = "Biopsie ou seconde opinion nécessaire"

        return {
            'confidence_score': float(confidence_score),
            'confidence_level': confidence_level,
            'confidence_color': confidence_color,
            'recommendation': recommendation,
            'sparsity': float(sparsity),
            'consensus_strength': float(consensus_strength),
            'avg_agreement': float(avg_agreement),
            'correlations': correlations
        }

    def apply_heatmap(self, image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
        h, w = image.shape[:2]
        h_resized = cv2.resize(heatmap, (w, h))
        h_colored = cv2.applyColorMap(np.uint8(255 * h_resized), colormap)
        h_colored = cv2.cvtColor(h_colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_norm = image.astype(np.float32) / 255.0 if image.max() > 1.0 else image.astype(np.float32)
        overlay = h_colored * alpha + image_norm * (1 - alpha)
        return np.clip(overlay, 0, 1)


# =======================================================
# PLATEFORME MÉDICALE COMPLÈTE (VRAI MODÈLE UNIQUEMENT)
# =======================================================

class UltimateMedicalPlatform:
    def __init__(self, model_path):
        print("\n" + "=" * 70)
        print("🚀 PLATEFORME MEDICALE ULTIME - Multi-XAI")
        print("   📄 Rapports PDF + 🎨 Visualisations 3D")
        print("=" * 70)

        # ===== VÉRIFICATION STRICTE DU MODÈLE =====
        print(f"🔍 Recherche du modèle: {model_path}")

        if not os.path.exists(model_path):
            print(f"❌ ERREUR CRITIQUE: Fichier modèle non trouvé!")
            print(f"   Chemin: {model_path}")
            print(f"   Répertoire actuel: {os.getcwd()}")

            # Liste des fichiers dans le dossier parent
            model_dir = os.path.dirname(model_path)
            if os.path.exists(model_dir):
                print(f"📂 Contenu de {model_dir}:")
                for f in os.listdir(model_dir):
                    print(f"   - {f}")
            else:
                print(f"📁 Le dossier {model_dir} n'existe pas")
                print(f"💡 Création du dossier models/...")
                os.makedirs(model_dir, exist_ok=True)

            raise FileNotFoundError(
                f"Modèle 'model.h5' non trouvé dans {model_dir}. "
                f"Placez votre modèle dans ce dossier."
            )

        # ===== CHARGEMENT DU VRAI MODÈLE =====
        print(f"📦 Chargement du modèle TensorFlow...")
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ Modèle chargé avec succès!")
            print(f"   Nom: {self.model.name}")
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            print(f"💡 Vérifiez que le fichier est un modèle Keras valide (.h5)")
            raise

        # ===== ANALYSE DE L'ARCHITECTURE =====
        inputs = self.model.inputs
        if isinstance(inputs, list) and len(inputs) == 1 and isinstance(inputs[0], list):
            inputs = inputs[0]

        image_input = inputs[0]
        self.IMG_SIZE = (int(image_input.shape[1]), int(image_input.shape[2]))
        self.is_multi_input = len(inputs) > 1

        if self.is_multi_input:
            self.NUM_METADATA = int(inputs[1].shape[1])
        else:
            self.NUM_METADATA = 0

        print(f"   Taille image: {self.IMG_SIZE}")
        print(f"   Multi-input: {self.is_multi_input} ({self.NUM_METADATA} métadonnées)")

        # Initialiser le moteur XAI
        self.xai_engine = UltimateXAIEngine(self.model, self.IMG_SIZE, self.is_multi_input, self.NUM_METADATA)

        # Initialiser les nouvelles fonctionnalités
        self.pdf_generator = MedicalPDFGenerator()
        self.visualizer_3d = XAI3DVisualizer()

        # Statistiques
        self.total_analyses = 0
        self.successful_analyses = 0
        print("=" * 70 + "\n")

    def preprocess_image(self, image_path):
        """Prétraitement de l'image"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError("Image non lisible")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.IMG_SIZE)
        img_norm = img_resized.astype(np.float32) / 255.0
        return np.expand_dims(img_norm, axis=0), img_rgb

    def preprocess_metadata(self, metadata_dict):
        """Prétraitement des métadonnées"""
        if not self.is_multi_input:
            return np.zeros((1, 0), dtype=np.float32)

        meta = np.zeros((1, self.NUM_METADATA), dtype=np.float32)
        age = float(metadata_dict.get('age', 45))
        meta[0, 0] = age / 100.0

        sex = metadata_dict.get('sex', 'F').upper()
        meta[0, 1] = 1.0 if sex == 'M' else 0.0

        loc_idx = int(metadata_dict.get('localization', 0))
        if 2 <= loc_idx + 2 < self.NUM_METADATA:
            meta[0, loc_idx + 2] = 1.0

        return meta

    def explain_metadata_with_lime(self, image_norm, metadata_vector, target_class_idx):
        """Explication LIME pour métadonnées"""
        try:
            import lime
            import lime.lime_tabular

            if not self.is_multi_input or self.NUM_METADATA == 0:
                return None, "Métadonnées non activées"

            # Fonction de prédiction pour LIME (UTILISE LE VRAI MODÈLE)
            def predict_for_lime(metadata_batch):
                batch_size = metadata_batch.shape[0]
                image_batch = np.repeat(image_norm[np.newaxis, :, :, :], batch_size, axis=0)
                inputs = [image_batch, metadata_batch]
                return self.model.predict(inputs, verbose=0)

            # Noms des features
            feature_names = ['Age', 'Sexe (M)']
            for i in range(2, self.NUM_METADATA):
                if i - 2 == 0:
                    feature_names.append('Localisation: Tete')
                elif i - 2 == 1:
                    feature_names.append('Localisation: Tronc')
                elif i - 2 == 2:
                    feature_names.append('Localisation: Membres')
                elif i - 2 == 3:
                    feature_names.append('Localisation: Mains/Pieds')
                else:
                    feature_names.append(f'Localisation_{i - 2}')

            # Données de fond
            background_data = np.zeros((50, self.NUM_METADATA))
            background_data[:, 0] = np.linspace(0.2, 0.8, 50)
            background_data[:, 1] = np.random.choice([0., 1.], 50)
            if self.NUM_METADATA > 2:
                background_data[:, 2] = 1.0

            # Expliquer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=background_data,
                feature_names=feature_names,
                class_names=CLASS_NAMES,
                categorical_features=list(range(2, min(self.NUM_METADATA, 6))),
                mode='classification',
                verbose=False,
                random_state=42
            )

            exp = explainer.explain_instance(
                data_row=metadata_vector,
                predict_fn=predict_for_lime,
                num_features=min(6, self.NUM_METADATA),
                top_labels=1,
                num_samples=300
            )

            # Visualisation
            fig, ax = plt.subplots(figsize=(8, 4))

            # Récupérer les explications
            exp_list = exp.as_list(label=target_class_idx)
            features = [x[0] for x in exp_list[:6]]
            values = [x[1] for x in exp_list[:6]]

            # Couleurs selon l'impact
            colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]

            ax.barh(range(len(features)), values, color=colors)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize=9)
            ax.set_xlabel("Contribution à la prédiction", fontweight='bold')
            ax.set_title(f"Impact des Métadonnées - {CLASS_NAMES[target_class_idx]}",
                         fontsize=12, fontweight='bold', pad=15)

            # Ligne verticale à 0
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            buf.seek(0)

            # Texte d'interprétation
            interpretation = "Principaux facteurs influençant la prédiction :\n"
            for feat, val in exp_list[:3]:
                direction = "augmente" if val > 0 else "diminue"
                interpretation += f"• {feat}: {direction} la probabilité ({val:+.3f})\n"

            return base64.b64encode(buf.getvalue()).decode('utf-8'), interpretation

        except ImportError:
            return None, "LIME non installé. pip install lime"
        except Exception as e:
            print(f"⚠️ Erreur LIME: {e}")
            return None, f"Erreur d'explication: {str(e)}"

    def predict_with_ultimate_xai(self, image_path, metadata_dict=None):
        """Prédiction complète avec XAI multi-techniques (VRAI MODÈLE)"""
        try:
            self.total_analyses += 1
            analysis_id = self.total_analyses
            print(f"\n{'=' * 70}")
            print(f"🔬 ANALYSE #{analysis_id} - XAI MULTI-TECHNIQUES")
            print(f"📁 Image: {Path(image_path).name}")
            print('=' * 70)

            # Prétraitement
            img_batch, original_rgb = self.preprocess_image(image_path)
            image_norm = img_batch[0]
            metadata = self.preprocess_metadata(metadata_dict or {})

            # ===== PRÉDICTION AVEC LE VRAI MODÈLE =====
            inputs = [img_batch, metadata] if self.is_multi_input else img_batch
            predictions = self.model.predict(inputs, verbose=0)[0]

            class_idx = int(np.argmax(predictions))
            pred_class = CLASS_NAMES[class_idx]
            confidence = float(predictions[class_idx])

            print(f"📊 Prédiction: {pred_class} ({confidence:.1%})")

            # ===== GÉNÉRATION DES CARTES XAI =====
            occlusion_map = self.xai_engine.occlusion_sensitivity(image_norm, metadata[0], class_idx)
            intgrad_map = self.xai_engine.integrated_gradients(image_norm, metadata[0], class_idx)
            attention_map = self.xai_engine.attention_rollout(image_norm, metadata[0])
            consensus_map = self.xai_engine.compute_consensus(occlusion_map, intgrad_map, attention_map)

            # Métriques de confiance
            confidence_metrics = self.xai_engine.compute_confidence_metrics(
                occlusion_map, intgrad_map, attention_map
            )

            print(
                f"📈 Confiance: {confidence_metrics['confidence_level']} ({confidence_metrics['confidence_score']:.2f})")
            print(f"🤝 Consensus: {confidence_metrics['consensus_strength']:.2f}")
            print(f"📊 Accord méthodes: {confidence_metrics['avg_agreement']:.2f}")

            # LIME pour métadonnées
            lime_b64, lime_interpretation = self.explain_metadata_with_lime(
                image_norm, metadata[0], class_idx
            )

            # Overlays
            overlay_occlusion = self.xai_engine.apply_heatmap(original_rgb, occlusion_map, alpha=0.5)
            overlay_intgrad = self.xai_engine.apply_heatmap(original_rgb, intgrad_map, alpha=0.5)
            overlay_attention = self.xai_engine.apply_heatmap(original_rgb, attention_map, alpha=0.5)
            overlay_consensus = self.xai_engine.apply_heatmap(original_rgb, consensus_map, alpha=0.6)

            # ===== GÉNÉRATION DES VISUALISATIONS 3D =====
            print("🎨 Génération des visualisations 3D...")

            # Préparer l'image pour les visualisations 3D
            img_for_3d = original_rgb.copy()
            if img_for_3d.max() > 1:
                img_for_3d = img_for_3d / 255.0

            # Générer les visualisations - utiliser HTML au lieu de JSON pour compatibilité
            viz_3d_multi_html = self.visualizer_3d.create_multi_3d_view_html(
                occlusion_map, intgrad_map, attention_map, consensus_map, img_for_3d
            )

            viz_3d_occlusion_html = self.visualizer_3d.create_3d_surface_html(
                occlusion_map, img_for_3d,
                title=f"Carte Occlusion 3D - {pred_class}",
                colorscale='Hot'
            )

            viz_3d_consensus_html = self.visualizer_3d.create_3d_surface_html(
                consensus_map, img_for_3d,
                title=f"Carte Consensus 3D - {pred_class}",
                colorscale='Viridis'
            )

            # ===== GÉNÉRATION DU RAPPORT PDF =====
            print("📄 Génération du rapport PDF...")

            # Préparer les données pour le PDF
            analysis_data = {
                'analysis_id': analysis_id,
                'predicted_class': pred_class,
                'confidence': confidence,
                'probabilities': {CLASS_NAMES[i]: float(p) for i, p in enumerate(predictions)},
                'confidence_metrics': confidence_metrics,
                'xai_metrics': {
                    'occlusion_max': float(occlusion_map.max()),
                    'intgrad_max': float(intgrad_map.max()),
                    'attention_max': float(attention_map.max()),
                    'consensus_max': float(consensus_map.max()),
                    'correlations': confidence_metrics['correlations'],
                    'metadata_available': self.is_multi_input
                },
                'lime_interpretation': lime_interpretation,
                'xai_methods': [
                    'Occlusion Sensitivity',
                    'Integrated Gradients',
                    'Attention Rollout',
                    'LIME (Métadonnées)',
                    'Métriques de Confiance'
                ],
                'timestamp': datetime.now().isoformat(),
                'user_info': {
                    'email': getattr(current_user, 'email', 'Non connecté'),
                    'full_name': getattr(current_user, 'full_name', 'Non spécifié'),
                    'institution': getattr(current_user, 'institution', 'Non spécifié')
                } if current_user.is_authenticated else None
            }

            # Générer le PDF
            pdf_result = self.pdf_generator.create_pdf_report(
                analysis_data,
                patient_info=metadata_dict,
                doctor_info={
                    'name': getattr(current_user, 'full_name', 'Dr. Non Spécifié'),
                    'institution': getattr(current_user, 'institution', 'Centre Médical'),
                    'rpps': '12345678901'  # Exemple
                }
            )

            # Dashboard complet
            dashboard_b64 = self.generate_professional_dashboard(
                original_rgb, predictions, class_idx,
                occlusion_map, intgrad_map, attention_map,
                consensus_map, confidence_metrics, metadata_dict or {}
            )

            # Encodage base64
            original_b64 = self.img_to_base64(original_rgb)
            overlay_occ_b64 = self.img_to_base64(overlay_occlusion)
            overlay_ig_b64 = self.img_to_base64(overlay_intgrad)
            overlay_att_b64 = self.img_to_base64(overlay_attention)
            overlay_consensus_b64 = self.img_to_base64(overlay_consensus)

            self.successful_analyses += 1
            print(f"\n✅ Analyse #{analysis_id} terminée avec succès!")
            print(f"🎯 Confiance: {confidence_metrics['confidence_level']}")
            print(f"📄 Rapport: {pdf_result['filename']}")
            print("=" * 70 + "\n")

            return {
                'success': True,
                'analysis_id': analysis_id,
                'predicted_class': pred_class,
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.1f}%",
                'confidence_metrics': confidence_metrics,
                'probabilities': {CLASS_NAMES[i]: float(p) for i, p in enumerate(predictions)},
                'original_image': original_b64,
                'overlay_occlusion': overlay_occ_b64,
                'overlay_integrated_gradients': overlay_ig_b64,
                'overlay_attention': overlay_att_b64,
                'overlay_consensus': overlay_consensus_b64,
                'dashboard_complete': dashboard_b64,
                'lime_metadata': lime_b64,
                'lime_interpretation': lime_interpretation,
                'xai_metrics': {
                    'occlusion_max': float(occlusion_map.max()),
                    'intgrad_max': float(intgrad_map.max()),
                    'attention_max': float(attention_map.max()),
                    'consensus_max': float(consensus_map.max()),
                    'correlations': confidence_metrics['correlations'],
                    'metadata_available': self.is_multi_input
                },
                'timestamp': datetime.now().isoformat(),
                'user_info': {
                    'email': getattr(current_user, 'email', 'Non connecté'),
                    'institution': getattr(current_user, 'institution', 'Non spécifié')
                } if current_user.is_authenticated else None,
                'xai_methods': [
                    'Occlusion Sensitivity',
                    'Integrated Gradients',
                    'Attention Rollout',
                    'LIME (Métadonnées)',
                    'Métriques de Confiance'
                ],
                'analysis_complete': True,

                # CORRECTION : Retourner du HTML au lieu de JSON
                'viz_3d_multi_html': viz_3d_multi_html,
                'viz_3d_occlusion_html': viz_3d_occlusion_html,
                'viz_3d_consensus_html': viz_3d_consensus_html,
                'pdf_report_url': pdf_result['url'],
                'pdf_filename': pdf_result['filename']
            }

        except Exception as e:
            import traceback
            print(f"❌ Erreur analyse: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'analysis_id': self.total_analyses
            }

    def generate_professional_dashboard(self, original_rgb, predictions, pred_class_idx,
                                        occlusion, intgrad, attention, consensus,
                                        confidence_metrics, metadata_dict):
        """Génère le dashboard professionnel"""
        fig = plt.figure(figsize=(22, 18), dpi=100)
        fig.patch.set_facecolor('#f8f9fa')

        img_display = original_rgb / 255.0 if original_rgb.max() > 1.0 else original_rgb
        pred_class = CLASS_NAMES[pred_class_idx]
        confidence = predictions[pred_class_idx]

        # ===== LIGNE 1: Diagnostic & Métadonnées =====
        ax1 = plt.subplot(5, 5, 1)
        ax1.imshow(img_display)
        ax1.set_title('IMAGE ORIGINALE', fontsize=11, fontweight='bold', pad=8)
        ax1.axis('off')

        # Top 5 prédictions
        ax2 = plt.subplot(5, 5, 2)
        top5_idx = np.argsort(predictions)[-5:][::-1]
        top5_names = [CLASS_NAMES[i] for i in top5_idx]
        top5_probs = predictions[top5_idx]
        colors_bar = [CLASS_COLORS[CLASS_NAMES[i]] for i in top5_idx]

        bars = ax2.barh(range(5), top5_probs, color=colors_bar, height=0.6, edgecolor='black')
        ax2.set_yticks(range(5))
        ax2.set_yticklabels(top5_names, fontweight='bold')
        ax2.set_xlim([0, 1])
        ax2.set_title('TOP 5 PREDICTIONS', fontsize=11, fontweight='bold', pad=8)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')

        for i, prob in enumerate(top5_probs):
            ax2.text(prob + 0.01, i, f'{prob:.1%}', va='center', fontweight='bold', fontsize=9)

        # Métadonnées
        ax3 = plt.subplot(5, 5, 3)
        ax3.axis('off')
        meta_text = f"""METADONNEES CLINIQUES
────────────────────
• Age: {metadata_dict.get('age', 'N/A')} ans
• Sexe: {metadata_dict.get('sex', 'N/A')}
• Localisation: {metadata_dict.get('localization', 'N/A')}

STATISTIQUES
• Analyse #{self.total_analyses}
• Taille: {self.IMG_SIZE[0]}x{self.IMG_SIZE[1]}
• Heure: {datetime.now().strftime('%H:%M:%S')}"""
        ax3.text(0.05, 0.5, meta_text, fontsize=9, fontfamily='monospace',
                 verticalalignment='center', color='#2c3e50',
                 bbox=dict(boxstyle='round', facecolor='#ecf0f1',
                           edgecolor='#bdc3c7', linewidth=1))

        # Score de confiance
        ax4 = plt.subplot(5, 5, 4)
        conf_score = confidence_metrics['confidence_score']
        conf_level = confidence_metrics['confidence_level']
        conf_color = {'ELEVE': '#28a745', 'MODERE': '#ffc107', 'LIMITE': '#dc3545'}.get(conf_level, '#6c757d')

        # Jauge de confiance
        ax4.barh([0], [conf_score], color=conf_color, height=0.3, edgecolor='black')
        ax4.set_xlim([0, 1])
        ax4.set_yticks([])
        ax4.set_title(f'NIVEAU DE CONFIANCE: {conf_level}',
                      fontsize=11, fontweight='bold', pad=8, color=conf_color)
        ax4.text(0.5, 0, f'{conf_score:.2f}', ha='center', va='center',
                 fontsize=14, fontweight='bold', color='white')
        ax4.grid(axis='x', alpha=0.3)

        # Recommandation
        ax5 = plt.subplot(5, 5, 5)
        ax5.axis('off')
        reco = confidence_metrics['recommendation']
        ax5.text(0.5, 0.5, f"RECOMMANDATION\n{reco}",
                 ha='center', va='center', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='#f8f9fa',
                           edgecolor=conf_color, linewidth=2))

        # ===== LIGNE 2: Occlusion Sensitivity =====
        ax6 = plt.subplot(5, 5, 6)
        ax6.imshow(occlusion, cmap='jet')
        ax6.set_title('OCCLUSION SENSITIVITY', fontsize=10, fontweight='bold', pad=6)
        ax6.axis('off')

        ax7 = plt.subplot(5, 5, 7)
        overlay_occ = self.xai_engine.apply_heatmap(img_display, occlusion, alpha=0.5)
        ax7.imshow(overlay_occ)
        ax7.set_title('ZONES CRITIQUES', fontsize=10, fontweight='bold', pad=6)
        ax7.axis('off')

        # ===== LIGNE 3: Integrated Gradients =====
        ax8 = plt.subplot(5, 5, 11)
        ax8.imshow(intgrad, cmap='jet')
        ax8.set_title('INTEGRATED GRADIENTS', fontsize=10, fontweight='bold', pad=6)
        ax8.axis('off')

        ax9 = plt.subplot(5, 5, 12)
        overlay_ig = self.xai_engine.apply_heatmap(img_display, intgrad, alpha=0.5)
        ax9.imshow(overlay_ig)
        ax9.set_title('ATTRIBUTION PIXELS', fontsize=10, fontweight='bold', pad=6)
        ax9.axis('off')

        # ===== LIGNE 4: Attention Rollout =====
        ax10 = plt.subplot(5, 5, 16)
        ax10.imshow(attention, cmap='jet')
        ax10.set_title('ATTENTION ROLLOUT', fontsize=10, fontweight='bold', pad=6)
        ax10.axis('off')

        ax11 = plt.subplot(5, 5, 17)
        overlay_att = self.xai_engine.apply_heatmap(img_display, attention, alpha=0.5)
        ax11.imshow(overlay_att)
        ax11.set_title('FOCUS DU MODELE', fontsize=10, fontweight='bold', pad=6)
        ax11.axis('off')

        # ===== LIGNE 5: Métriques & Consensus =====
        # Comparaison méthodes
        ax12 = plt.subplot(5, 5, 8)
        methods = ['Occlusion', 'Int.Grad', 'Attention']
        max_vals = [occlusion.max(), intgrad.max(), attention.max()]
        mean_vals = [occlusion.mean(), intgrad.mean(), attention.mean()]

        x = np.arange(len(methods))
        width = 0.35

        bars1 = ax12.bar(x - width / 2, max_vals, width, label='Maximum', color='#e74c3c', alpha=0.8)
        bars2 = ax12.bar(x + width / 2, mean_vals, width, label='Moyenne', color='#3498db', alpha=0.8)

        ax12.set_ylabel('Valeur', fontweight='bold')
        ax12.set_title('COMPARAISON METHODES', fontsize=10, fontweight='bold', pad=6)
        ax12.set_xticks(x)
        ax12.set_xticklabels(methods)
        ax12.legend(fontsize=8)
        ax12.grid(axis='y', alpha=0.3, linestyle='--')

        # Corrélations
        ax13 = plt.subplot(5, 5, 9)
        corr_values = list(confidence_metrics['correlations'].values())
        corr_labels = ['Occ-IG', 'Occ-Att', 'IG-Att']

        bars_corr = ax13.bar(corr_labels, corr_values, color=['#2ecc71', '#f39c12', '#9b59b6'])
        ax13.set_ylim([-1, 1])
        ax13.set_title('CORRELATIONS', fontsize=10, fontweight='bold', pad=6)
        ax13.axhline(y=0, color='black', linewidth=0.8)
        ax13.grid(axis='y', alpha=0.3, linestyle='--')

        for i, v in enumerate(corr_values):
            ax13.text(i, v + (0.05 if v >= 0 else -0.08), f'{v:.2f}',
                      ha='center', fontweight='bold', fontsize=9)

        # Métriques de qualité
        ax14 = plt.subplot(5, 5, 10)
        ax14.axis('off')

        metrics_text = f"""METRIQUES DE QUALITE
────────────────────
Consensus: {confidence_metrics['consensus_strength']:.2f}
Accord methodes: {confidence_metrics['avg_agreement']:.2f}
Sparsité: {confidence_metrics['sparsity']:.2f}

SCORE COMPOSITE
{conf_score:.2f} / 1.00

NIVEAU
{conf_level}"""

        ax14.text(0.05, 0.5, metrics_text, fontsize=9, fontfamily='monospace',
                  verticalalignment='center', color='#2c3e50',
                  bbox=dict(boxstyle='round', facecolor='#ecf0f1',
                            edgecolor='#bdc3c7', linewidth=1))

        # Consensus map
        ax15 = plt.subplot(5, 5, 13)
        im = ax15.imshow(consensus, cmap='viridis')
        ax15.set_title('CONSENSUS 3 METHODES', fontsize=10, fontweight='bold', pad=6)
        ax15.axis('off')
        plt.colorbar(im, ax=ax15, fraction=0.046, pad=0.04)

        # Overlay consensus
        ax16 = plt.subplot(5, 5, 14)
        overlay_cons = self.xai_engine.apply_heatmap(img_display, consensus, alpha=0.6, colormap=cv2.COLORMAP_VIRIDIS)
        ax16.imshow(overlay_cons)
        ax16.set_title('ZONES D ACCORD', fontsize=10, fontweight='bold', pad=6)
        ax16.axis('off')

        # Interprétation finale
        ax17 = plt.subplot(5, 5, 15)
        ax17.axis('off')

        interp_text = f"""INTERPRETATION CLINIQUE
──────────────────────
Diagnostic: {pred_class}
Confiance prédiction: {confidence:.1%}
Niveau confiance XAI: {conf_level}

FACTEURS CLES:
• Consensus fort: {confidence_metrics['consensus_strength']:.2f}
• Accord methodes: {confidence_metrics['avg_agreement']:.2f}
• Explication focalisée: {confidence_metrics['sparsity']:.2f}

CLINIQUE:
{confidence_metrics['recommendation']}"""

        ax17.text(0.05, 0.5, interp_text, fontsize=9, fontfamily='monospace',
                  verticalalignment='center', color='#2c3e50',
                  bbox=dict(boxstyle='round', facecolor='#d5f4e6' if conf_level == 'ELEVE'
                  else '#fff3cd' if conf_level == 'MODERE' else '#f8d7da',
                            edgecolor='#28a745' if conf_level == 'ELEVE'
                            else '#ffc107' if conf_level == 'MODERE' else '#dc3545',
                            linewidth=2))

        plt.suptitle(f'PLATEFORME MEDICALE ULTIME - ANALYSE #{self.total_analyses}\n'
                     f'Diagnostic: {pred_class} ({confidence:.1%}) | Confiance XAI: {conf_level} | {datetime.now().strftime("%d/%m/%Y %H:%M")}',
                     fontsize=14, fontweight='bold', color='#2c3e50')

        plt.tight_layout(rect=[0, 0.02, 1, 0.98])

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close(fig)
        buf.seek(0)

        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def img_to_base64(self, img_array):
        """Convertir image en base64"""
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)

        img_pil = Image.fromarray(img_array)
        buf = io.BytesIO()
        img_pil.save(buf, format='PNG', optimize=True)
        return base64.b64encode(buf.getvalue()).decode('utf-8')


# =======================================================
# INITIALISATION PLATEFORME (VÉRIFICATION STRICTE)
# =======================================================

MODEL_PATH = BASE_DIR / "models" / "model.h5"

print(f"🔍 Vérification du chemin du modèle...")
print(f"   BASE_DIR: {BASE_DIR}")
print(f"   Chemin modèle: {MODEL_PATH}")

# Vérification stricte - PAS DE SIMULATION
if not MODEL_PATH.exists():
    print(f"\n❌ ERREUR CRITIQUE: Modèle 'model.h5' non trouvé!")
    print(f"   Chemin recherché: {MODEL_PATH}")
    print(f"\n💡 SOLUTIONS:")
    print(f"1. Placez votre fichier model.h5 dans: {MODEL_PATH.parent}")
    print(f"2. OU modifiez le chemin dans le code:")
    print(f"   MODEL_PATH = BASE_DIR / 'votre_dossier' / 'votre_modele.h5'")

    # Créer le dossier pour aider l'utilisateur
    MODEL_PATH.parent.mkdir(exist_ok=True)
    print(f"\n📁 Dossier créé: {MODEL_PATH.parent}")
    print(f"   Placez-y votre modèle et relancez l'application.")

    sys.exit(1)

print(f"✅ Modèle trouvé: {MODEL_PATH}")
platform = UltimateMedicalPlatform(str(MODEL_PATH))

# Créer la base de données
with app.app_context():
    db.create_all()
    print("✅ Base de données initialisée")


# =======================================================
# ROUTES AUTHENTIFICATION
# =======================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user)
            return jsonify({'success': True, 'redirect': '/dashboard'})
        else:
            return jsonify({'success': False, 'error': 'Email ou mot de passe incorrect'})

    return render_template('medical_login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        full_name = request.form.get('full_name')
        institution = request.form.get('institution')
        password = request.form.get('password')

        # Vérifier si l'utilisateur existe
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'error': 'Cet email est déjà utilisé'})

        # Créer nouvel utilisateur
        new_user = User(
            email=email,
            full_name=full_name,
            institution=institution,
            role='medecin'
        )
        new_user.set_password(password)

        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)
        return jsonify({'success': True, 'redirect': '/dashboard'})

    return render_template('medical_signup.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return jsonify({'success': True, 'redirect': '/'})


# =======================================================
# ROUTES PRINCIPALES
# =======================================================
@app.context_processor
def inject_globals():
    """Injecte des variables globales dans tous les templates"""
    return {
        'app_name': 'Plateforme Médicale XAI',
        'app_version': '2.0',
        'current_year': datetime.now().year,
        'classes': CLASS_NAMES,
        'class_display_names': {
            'AKIEC': 'Kératose Actinique',
            'BCC': 'Carcinome Basocellulaire',
            'BKL': 'Lésion Bénigne de la Kératose',
            'DF': 'Dermatofibrome',
            'MEL': 'Mélanome',
            'NV': 'Naevus Mélanocytaire',
            'VASC': 'Lésion Vasculaire',
            'SCC': 'Carcinome Spinocellulaire'
        },
        'class_colors': CLASS_COLORS
    }
@app.route('/')
def index():
    """Page d'accueil publique"""
    # Compter les utilisateurs actifs
    active_users = User.query.count()

    # Statistiques pour la page d'accueil
    stats = {
        'total_analyses': platform.total_analyses,
        'successful_analyses': platform.successful_analyses,
        'model_name': getattr(platform.model, 'name', 'Modèle Médical XAI'),
        'active_users': active_users
    }

    return render_template('medical_index.html',
                           stats=stats,
                           classes=CLASS_NAMES,
                           app_name='Plateforme Médicale XAI',
                           app_version='2.0',
                           current_year=datetime.now().year,
                           user=current_user)

@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard principal - page d'accueil après connexion"""
    model_info = {
        'name': getattr(platform.model, 'name', 'MultiInput_DenseNet_Medical'),
        'img_size': platform.IMG_SIZE,
        'multi_input': platform.is_multi_input,
        'metadata_features': platform.NUM_METADATA,
        'total_analyses': platform.total_analyses,
        'successful_analyses': platform.successful_analyses,
        'model_status': '✅ Opérationnel (Vrai modèle)',
        'last_update': datetime.now().strftime("%d/%m/%Y %H:%M")
    }

    # Récupérer analyses récentes
    recent_files = list(RESULTS_FOLDER.glob("*.png"))
    recent_analyses = []
    if recent_files:
        recent_files.sort(key=os.path.getmtime, reverse=True)
        for file in recent_files[:5]:
            # CORRECTION ICI : Créer un dictionnaire avec l'attribut created_at
            created_at = datetime.fromtimestamp(os.path.getmtime(file))
            recent_analyses.append({
                'filename': file.name,
                'time': created_at.strftime("%H:%M"),
                'date': created_at.strftime("%d/%m/%Y"),
                'created_at': created_at  # <-- AJOUTER CETTE LIGNE
            })

    return render_template(
        'medical_analysis_new.html',
        model_info=model_info,
        recent_analyses=recent_analyses,  # Maintenant chaque élément a 'created_at'
        classes=CLASS_NAMES,
        user=current_user,
        current_year=datetime.now().year,
        app_name='Plateforme Médicale XAI'
    )


# Ajoutez en haut du fichier, après les imports
from flask import session
import pickle

# Dictionnaire global pour stocker les résultats (en production, utilisez Redis ou base de données)
analysis_results = {}


@app.route('/api/predict_ultimate', methods=['POST'])
@login_required
def predict_ultimate():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier'}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': 'Fichier invalide'}), 400

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = UPLOAD_FOLDER / f"{timestamp}_{filename}"
        file.save(str(filepath))

        metadata = {}
        if request.form.get('age'):
            metadata['age'] = float(request.form['age'])
        if request.form.get('sex'):
            metadata['sex'] = request.form['sex']
        if request.form.get('localization'):
            metadata['localization'] = int(request.form['localization'])

        # Exécuter l'analyse
        result = platform.predict_with_ultimate_xai(str(filepath), metadata)

        # Stocker le résultat dans le dictionnaire global
        analysis_id = result['analysis_id']
        analysis_results[analysis_id] = result

        # Stocker également dans la session pour l'utilisateur courant
        session['last_analysis_id'] = analysis_id

        # Sauvegarder l'image originale
        if result.get('original_image'):
            img_data = base64.b64decode(result['original_image'])
            img_path = RESULTS_FOLDER / f"analysis_{analysis_id}_original.png"
            with open(img_path, 'wb') as f:
                f.write(img_data)

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/analysis/<analysis_id>')
@login_required
def view_analysis(analysis_id):
    """Page de visualisation des résultats d'analyse avec VRAIES données"""

    # Récupérer les données de l'analyse depuis le stockage
    result = analysis_results.get(int(analysis_id))

    if not result:
        # Si non trouvé, tenter de récupérer le dernier de la session
        last_id = session.get('last_analysis_id')
        if last_id:
            result = analysis_results.get(int(last_id))
            if result:
                analysis_id = last_id

    if not result:
        return render_template('analysis_not_found.html', analysis_id=analysis_id), 404

    # Préparer les données pour le template
    analysis_data = {
        'analysis_id': analysis_id,
        'predicted_class': result['predicted_class'],
        'confidence': result['confidence'],
        'confidence_percent': result['confidence_percent'],
        'confidence_metrics': result['confidence_metrics'],
        'probabilities': result['probabilities'],
        'xai_metrics': result['xai_metrics'],
        'timestamp': result['timestamp'],
        'created_at': datetime.fromisoformat(result['timestamp']),
        'lime_interpretation': result.get('lime_interpretation', ''),
        'xai_methods': result.get('xai_methods', []),
        'patient_age': request.args.get('age', 'N/A'),
        'patient_sex': request.args.get('sex', 'N/A'),
        'patient_localization': request.args.get('localization', 'N/A'),
        'validated_by_doctor': False,
        'clinical_feedback': None,
        'feedback_rating': 0,
        'original_image': result.get('original_image'),
        'overlay_occlusion': result.get('overlay_occlusion'),
        'overlay_integrated_gradients': result.get('overlay_integrated_gradients'),
        'overlay_attention': result.get('overlay_attention'),
        'overlay_consensus': result.get('overlay_consensus'),
        'dashboard_complete': result.get('dashboard_complete'),
        'viz_3d_multi_html': result.get('viz_3d_multi_html'),
        'viz_3d_occlusion_html': result.get('viz_3d_occlusion_html'),
        'viz_3d_consensus_html': result.get('viz_3d_consensus_html'),
        'pdf_report_url': result.get('pdf_report_url'),
        'pdf_filename': result.get('pdf_filename')
    }

    # Récupérer les métadonnées de la requête
    metadata = {
        'age': request.args.get('age'),
        'sex': request.args.get('sex'),
        'localization': request.args.get('localization')
    }

    # Charger les guidelines pour la pathologie détectée
    guidelines = load_guidelines_for_class(analysis_data['predicted_class'])

    return render_template(
        'medical_analysis_view.html',
        analysis=analysis_data,
        probabilities=analysis_data['probabilities'],
        confidence_metrics=analysis_data['confidence_metrics'],
        xai_metrics=analysis_data['xai_metrics'],
        metadata=metadata,
        guidelines=guidelines,
        user=current_user,
        current_year=datetime.now().year,
        app_name='Plateforme Médicale XAI',
        class_display_names={
            'AKIEC': 'Kératose Actinique',
            'BCC': 'Carcinome Basocellulaire',
            'BKL': 'Lésion Bénigne de la Kératose',
            'DF': 'Dermatofibrome',
            'MEL': 'Mélanome',
            'NV': 'Naevus Mélanocytaire',
            'VASC': 'Lésion Vasculaire',
            'SCC': 'Carcinome Spinocellulaire'
        },
        class_colors=CLASS_COLORS
    )


def load_guidelines_for_class(class_name):
    """Charger les guidelines pour une classe spécifique"""
    guidelines_db = {
        'AKIEC': {
            'title': 'Kératose Actinique',
            'description': 'Lésion précancéreuse due à l\'exposition solaire chronique',
            'risk_factors': ['Exposition solaire chronique', 'Âge > 50 ans', 'Peau claire', 'Immunosuppression'],
            'diagnostic_criteria': ['Lésion rugueuse au toucher', 'Érythème', 'Squames adhérentes',
                                    'Taille 2-6 mm'],
            'recommendations': ['Protection solaire stricte', 'Surveillance régulière', 'Traitement préventif'],
            'treatment_options': ['Cryothérapie', 'Crème 5-FU', 'Imiquimod', 'Photothérapie dynamique', 'Curetage'],
            'severity_level': 'moderate'
        },
        'BCC': {
            'title': 'Carcinome Basocellulaire',
            'description': 'Cancer cutané le plus fréquent, croissance lente, faible potentiel métastatique',
            'risk_factors': ['Exposition UV chronique', 'Peau claire', 'Âge > 40 ans', 'Antécédents familiaux'],
            'diagnostic_criteria': ['Papule perlée', 'Télangiectasies', 'Ulcération centrale', 'Bords enroulés'],
            'recommendations': ['Exérèse chirurgicale complète', 'Contrôle des marges', 'Suivi annuel'],
            'treatment_options': ['Chirurgie excisionnelle', 'Chirurgie de Mohs', 'Cryothérapie', 'Radiothérapie'],
            'severity_level': 'moderate'
        },
        'MEL': {
            'title': 'Mélanome',
            'description': 'Cancer cutané le plus grave, potentiel métastatique élevé',
            'risk_factors': ['Exposition UV intense intermittente', 'Nombreux naevi', 'Antécédents familiaux',
                             'Phénotype clair'],
            'diagnostic_criteria': [
                'ABCDE: Asymétrie, Bords irréguliers, Couleur hétérogène, Diamètre >6mm, Évolution'],
            'recommendations': ['URGENCE: Exérèse large', 'Biopsie ganglion sentinelle', 'Stadification complète'],
            'treatment_options': ['Exérèse large', 'Immunothérapie', 'Chimiothérapie', 'Radiothérapie'],
            'severity_level': 'high'
        },
        'NV': {
            'title': 'Naevus Mélanocytaire',
            'description': 'Grain de beauté bénin, transformation maligne rare',
            'risk_factors': ['Génétique', 'Exposition solaire', 'Peau claire'],
            'diagnostic_criteria': ['Symétrique', 'Bords réguliers', 'Couleur homogène', 'Stable dans le temps'],
            'recommendations': ['Auto-surveillance mensuelle', 'Photographie de référence',
                                'Dermatoscopie annuelle'],
            'treatment_options': ['Surveillance simple', 'Exérèse si modification'],
            'severity_level': 'low'
        },
        'SCC': {
            'title': 'Carcinome Spinocellulaire',
            'description': 'Cancer cutané agressif avec risque métastatique modéré',
            'risk_factors': ['Exposition UV chronique', 'Immunosuppression', 'HPV', 'Lésions précancéreuses'],
            'diagnostic_criteria': ['Nodule induré', 'Ulcération', 'Croissance rapide', 'Saignement'],
            'recommendations': ['Exérèse chirurgicale large', 'Examen des ganglions', 'Surveillance rapprochée'],
            'treatment_options': ['Chirurgie', 'Radiothérapie', 'Chimiothérapie si métastases'],
            'severity_level': 'moderate'
        }
    }

    return guidelines_db.get(class_name, {
        'title': class_name,
        'description': 'Pathologie cutanée nécessitant évaluation spécialisée',
        'recommendations': ['Consultation dermatologique', 'Biopsie si doute diagnostique']
    })


# =======================================================
# NOUVELLES ROUTES API
# =======================================================

@app.route('/api/download_pdf/<analysis_id>')
@login_required
def download_pdf(analysis_id):
    """Télécharger un rapport PDF"""
    try:
        # Rechercher le fichier PDF correspondant
        pdf_pattern = f"medical_report_{analysis_id}_*.pdf"
        pdf_files = list(REPORTS_FOLDER.glob(pdf_pattern))

        if not pdf_files:
            # Rechercher le dernier PDF généré
            pdf_files = list(REPORTS_FOLDER.glob("*.pdf"))
            pdf_files.sort(key=os.path.getmtime, reverse=True)

        if pdf_files:
            latest_pdf = pdf_files[0]
            return send_file(
                str(latest_pdf),
                as_attachment=True,
                download_name=f"rapport_medical_{analysis_id}_{datetime.now().strftime('%Y%m%d')}.pdf",
                mimetype='application/pdf'
            )
        else:
            return jsonify({'error': 'Rapport PDF non trouvé'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate_3d_viz', methods=['POST'])
@login_required
def generate_3d_viz():
    """Générer une visualisation 3D personnalisée"""
    try:
        data = request.json
        method = data.get('method', 'surface')

        if method == 'surface':
            # Générer une surface 3D simple
            heatmap = np.random.rand(50, 50)  # Exemple
            # Utiliser la version HTML pour compatibilité
            viz_html = platform.visualizer_3d.create_3d_surface_html(
                heatmap,
                title="Visualisation 3D Personnalisée"
            )
            return jsonify({'success': True, 'html': viz_html})

        else:
            return jsonify({'error': 'Méthode non supportée'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/user_info')
@login_required
def get_user_info():
    return jsonify({
        'email': current_user.email,
        'full_name': current_user.full_name,
        'institution': current_user.institution,
        'role': current_user.role,
        'created_at': current_user.created_at.isoformat() if current_user.created_at else None
    })


@app.route('/analysis_history')
@login_required
def analysis_history():
    """Historique des analyses de l'utilisateur"""
    # Récupérer tous les fichiers PDF de l'utilisateur
    pdf_files = list(REPORTS_FOLDER.glob("*.pdf"))
    pdf_files.sort(key=os.path.getmtime, reverse=True)

    analyses = []
    for pdf_file in pdf_files[:20]:  # Limiter aux 20 derniers
        analyses.append({
            'filename': pdf_file.name,
            'path': str(pdf_file),
            'date': datetime.fromtimestamp(os.path.getmtime(pdf_file)).strftime("%d/%m/%Y"),
            'time': datetime.fromtimestamp(os.path.getmtime(pdf_file)).strftime("%H:%M"),
            'analysis_id': pdf_file.stem.split('_')[2] if len(pdf_file.stem.split('_')) > 2 else 'N/A'
        })

    return render_template(
        'medical_analysis_history.html',
        analyses=analyses,
        user=current_user,
        current_year=datetime.now().year
    )


@app.route('/patient_management')
@login_required
def patient_management():
    """Gestion des patients (page vide pour le moment)"""
    return render_template(
        'medical_patient_management.html',
        user=current_user,
        current_year=datetime.now().year
    )


@app.route('/settings')
@login_required
def settings():
    """Paramètres utilisateur"""
    return render_template(
        'medical_settings.html',
        user=current_user,
        current_year=datetime.now().year
    )


@app.route('/new_analysis')
@login_required
def new_analysis():
    """Page pour faire une nouvelle analyse"""
    # Informations sur le modèle
    model_info = {
        'img_size': platform.IMG_SIZE,
        'multi_input': platform.is_multi_input,
        'metadata_features': platform.NUM_METADATA,
        'model_name': getattr(platform.model, 'name', 'Modèle Médical XAI'),
        'status': '✅ Opérationnel'
    }

    # Informations pour l'analyse
    analysis = {
        'status': 'new',
        'id': f"ANA_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'date': datetime.now().strftime("%d/%m/%Y %H:%M"),
        'predicted_class': None,
        'confidence': None,
        'user': {
            'name': current_user.full_name,
            'institution': current_user.institution,
            'email': current_user.email
        }
    }

    # Classes pour le template
    class_info = {
        'names': CLASS_NAMES,
        'colors': CLASS_COLORS,
        'descriptions': {
            'AKIEC': 'Kératose Actinique',
            'BCC': 'Carcinome Basocellulaire',
            'BKL': 'Lésion Bénigne de la Kératose',
            'DF': 'Dermatofibrome',
            'MEL': 'Mélanome',
            'NV': 'Naevus Mélanocytaire',
            'VASC': 'Lésion Vasculaire',
            'SCC': 'Carcinome Spinocellulaire'
        }
    }

    return render_template(
        'medical_analysis_new.html',
        user=current_user,
        analysis=analysis,
        model_info=model_info,
        classes=class_info,  # Passage des classes avec descriptions
        current_year=datetime.now().year,
        app_name='Plateforme Médicale XAI',
        app_version='2.0'
    )


@app.route('/medical_guidelines')
@login_required
def medical_guidelines():
    """Guidelines médicales par pathologie"""

    # Définir les guidelines pour chaque pathologie
    guidelines = {
        'AKIEC': {
            'name': 'Kératose Actinique',
            'description': 'Lésion précancéreuse due à l\'exposition solaire',
            'risk_factors': ['Exposition solaire chronique', 'Âge > 50 ans', 'Peau claire'],
            'diagnostic_criteria': ['Lésion rugueuse', 'Érythème', 'Squames adhérentes'],
            'treatment': ['Cryothérapie', 'Crèmes topiques (5-FU, Imiquimod)', 'Photothérapie dynamique'],
            'follow_up': 'Surveillance tous les 6 mois'
        },
        'BCC': {
            'name': 'Carcinome Basocellulaire',
            'description': 'Cancer cutané le plus fréquent, croissance lente',
            'risk_factors': ['Exposition UV', 'Peau claire', 'Immunosuppression'],
            'diagnostic_criteria': ['Papule perlée', 'Télangiectasies', 'Ulcération centrale'],
            'treatment': ['Exérèse chirurgicale', 'Chirurgie de Mohs', 'Radiothérapie'],
            'follow_up': 'Surveillance annuelle'
        },
        'BKL': {
            'name': 'Lésion Bénigne de la Kératose',
            'description': 'Lésion bénigne liée à l\'âge',
            'risk_factors': ['Âge', 'Prédisposition génétique'],
            'diagnostic_criteria': ['Surface verruqueuse', 'Aspect "collé"', 'Pigmentation variable'],
            'treatment': ['Surveillance', 'Cryothérapie si gênante', 'Curetage'],
            'follow_up': 'Pas de surveillance nécessaire'
        },
        'DF': {
            'name': 'Dermatofibrome',
            'description': 'Nodule fibreux bénin du derme',
            'risk_factors': ['Traumatisme cutané', 'Piqûre d\'insecte'],
            'diagnostic_criteria': ['Nodule ferme', 'Signe du fossette', 'Couleur brune'],
            'treatment': ['Surveillance', 'Exérèse si gênant'],
            'follow_up': 'Aucune surveillance requise'
        },
        'MEL': {
            'name': 'Mélanome',
            'description': 'Cancer cutané le plus grave, métastases rapides',
            'risk_factors': ['Antécédents familiaux', 'Nombreux naevi', 'Exposition UV intense'],
            'diagnostic_criteria': ['ABCDE: Asymétrie, Bords, Couleur, Diamètre, Évolution'],
            'treatment': ['URGENCE: Exérèse large', 'Biopsie du ganglion sentinelle', 'Immunothérapie/Chimiothérapie'],
            'follow_up': 'Surveillance rapprochée tous les 3 mois'
        },
        'NV': {
            'name': 'Naevus Mélanocytaire',
            'description': 'Grain de beauté bénin',
            'risk_factors': ['Génétique', 'Exposition solaire'],
            'diagnostic_criteria': ['Symétrique', 'Bords réguliers', 'Couleur homogène'],
            'treatment': ['Surveillance', 'Exérèse si modification'],
            'follow_up': 'Auto-surveillance mensuelle'
        },
        'VASC': {
            'name': 'Lésion Vasculaire',
            'description': 'Lésion bénigne des vaisseaux sanguins',
            'risk_factors': ['Congénital ou acquis', 'Traumatisme'],
            'diagnostic_criteria': ['Couleur rouge/violacée', 'Blanchit à la pression'],
            'treatment': ['Surveillance', 'Laser si esthétique', 'Sclérothérapie'],
            'follow_up': 'Surveillance si évolution'
        },
        'SCC': {
            'name': 'Carcinome Spinocellulaire',
            'description': 'Cancer cutané agressif avec risque métastatique',
            'risk_factors': ['Exposition UV chronique', 'Immunosuppression', 'HPV'],
            'diagnostic_criteria': ['Nodule induré', 'Ulcération', 'Croissance rapide'],
            'treatment': ['Exérèse chirurgicale large', 'Radiothérapie', 'Chimiothérapie si métastases'],
            'follow_up': 'Surveillance tous les 3-6 mois'
        }
    }

    return render_template(
        'medical_guidelines.html',
        guidelines=guidelines,
        user=current_user,
        current_year=datetime.now().year
    )


# =======================================================
# LANCEMENT
# =======================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🏥 PLATEFORME MEDICALE ULTIME PRÊTE")
    print("   XAI Multi-Techniques + LIME + Authentification")
    print("   📄 Rapports PDF + 🎨 Visualisations 3D (HTML CORRIGÉ)")
    print("   VRAI MODÈLE UNIQUEMENT - PAS DE SIMULATION")
    print("=" * 70)
    print(f"📁 Modèle: {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"📁 Uploads: {UPLOAD_FOLDER}")
    print(f"📁 Résultats: {RESULTS_FOLDER}")
    print(f"📁 Rapports: {REPORTS_FOLDER}")
    print(f"🏷️ Classes: {len(CLASS_NAMES)}")
    print(f"🔐 Authentification: Activée")
    print(f"🎯 Métriques confiance: Activées")
    print(f"📄 Génération PDF: Activée")
    print(f"🎨 Visualisations 3D: Activées (HTML)")
    print("=" * 70)
    print("🌐 Serveur: http://localhost:5000")
    print("👤 Compte test: test@medecin.fr / password")
    print("=" * 70)

    # Installer les dépendances manquantes
    print("\n📦 Vérification des dépendances...")
    try:
        import reportlab
        print("✅ ReportLab: OK")
    except ImportError:
        print("⚠️ ReportLab non installé. Exécutez: pip install reportlab")

    try:
        import plotly
        print("✅ Plotly: OK")
    except ImportError:
        print("⚠️ Plotly non installé. Exécutez: pip install plotly")

    try:
        import lime
        print("✅ LIME: OK")
    except ImportError:
        print("⚠️ LIME non installé. Exécutez: pip install lime")

    print("=" * 70)

    app.run(debug=True, host='0.0.0.0', port=5000)