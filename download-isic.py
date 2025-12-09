# download_isic_2019_official.py
"""
Script de téléchargement officiel ISIC 2019
Source: https://challenge.isic-archive.com/data/#2019
Dataset: 25,331 images + métadonnées pour classification de lésions cutanées
License: CC-BY-NC (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm

# Dossier de destination
DATA_DIR = Path("data/isic_2019")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 🔗 LIENS OFFICIELS AWS S3 (hébergement public gratuit par ISIC)
FILES = {
    "images": {
        "url": "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Input.zip",
        "dest": DATA_DIR / "ISIC_2019_Training_Input.zip",
        "size": "9.1 GB"
    },
    "ground_truth": {
        "url": "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_GroundTruth.csv",
        "dest": DATA_DIR / "ISIC_2019_Training_GroundTruth.csv",
        "size": "1 MB"
    },
    "metadata": {
        "url": "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Metadata.csv",
        "dest": DATA_DIR / "ISIC_2019_Training_Metadata.csv",
        "size": "1 MB"
    }
}


def format_bytes(size):
    """Convertit bytes en format lisible"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def download_file(url, dest, expected_size=""):
    """Télécharge un fichier avec barre de progression"""
    if dest.exists():
        print(f"✅ Déjà présent : {dest.name} ({format_bytes(dest.stat().st_size)})")
        return

    print(f"\n📥 Téléchargement de {dest.name} ({expected_size})...")
    print(f"   URL : {url}")

    try:
        # Requête avec stream
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        # Téléchargement avec barre de progression
        with open(dest, "wb") as f, tqdm(
                desc=dest.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        print(f"✅ Téléchargé : {dest.name} ({format_bytes(dest.stat().st_size)})")

    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur de téléchargement : {e}")
        if dest.exists():
            dest.unlink()  # Supprimer le fichier incomplet
        raise
    except Exception as e:
        print(f"❌ Erreur : {e}")
        if dest.exists():
            dest.unlink()
        raise


def extract_images(zip_path, extract_to):
    """Extrait les images du ZIP"""
    if extract_to.exists() and len(list(extract_to.glob("*.jpg"))) > 20000:
        jpg_count = len(list(extract_to.glob("*.jpg")))
        print(f"✅ Images déjà extraites : {jpg_count:,} fichiers dans {extract_to}")
        return

    print(f"\n📦 Extraction de {zip_path.name}...")
    extract_to.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Compter le nombre total de fichiers
            total_files = len(zf.namelist())

            # Extraire avec progression
            for file in tqdm(zf.namelist(), desc="Extraction", unit="fichier"):
                zf.extract(file, extract_to)

        print(f"✅ Extraction terminée")

        # Supprimer le ZIP pour économiser l'espace disque
        print(f"🗑️  Suppression du ZIP pour libérer l'espace...")
        zip_path.unlink()
        print(f"✅ ZIP supprimé")

    except Exception as e:
        print(f"❌ Erreur d'extraction : {e}")
        raise


def verify_dataset():
    """Vérifie l'intégrité du dataset"""
    print("\n🔍 Vérification du dataset...")

    img_dir = DATA_DIR / "ISIC_2019_Training_Input"
    jpg_count = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
    csv_gt = FILES["ground_truth"]["dest"].exists()
    csv_meta = FILES["metadata"]["dest"].exists()

    print(f"   Images        : {jpg_count:,} / 25,331 attendues")
    print(f"   Ground Truth  : {'✅' if csv_gt else '❌'} {FILES['ground_truth']['dest'].name}")
    print(f"   Metadata      : {'✅' if csv_meta else '❌'} {FILES['metadata']['dest'].name}")

    if jpg_count >= 25000 and csv_gt and csv_meta:
        print(f"\n🎉 Dataset ISIC 2019 prêt à l'emploi !")
        print(f"\n📊 Structure du dataset :")
        print(f"   - Images       : {img_dir}")
        print(f"   - Labels       : {FILES['ground_truth']['dest']}")
        print(f"   - Métadonnées  : {FILES['metadata']['dest']}")
        print(f"\n📚 Classes de lésions :")
        print(f"   MEL (Melanoma), NV (Nevus), BCC (Basal cell carcinoma),")
        print(f"   AK (Actinic keratosis), BKL (Benign keratosis),")
        print(f"   DF (Dermatofibroma), VASC (Vascular lesion), SCC (Squamous cell carcinoma)")
        return True
    else:
        print(f"\n⚠️  Dataset incomplet - Relancer le script")
        return False


def main():
    print("=" * 80)
    print("  TÉLÉCHARGEMENT ISIC 2019 - Classification de lésions cutanées")
    print("  Source : ISIC Archive (AWS S3 public)")
    print("  License: CC-BY-NC 4.0")
    print("=" * 80)

    try:
        # Télécharger les fichiers CSV (rapides)
        download_file(
            FILES["ground_truth"]["url"],
            FILES["ground_truth"]["dest"],
            FILES["ground_truth"]["size"]
        )

        download_file(
            FILES["metadata"]["url"],
            FILES["metadata"]["dest"],
            FILES["metadata"]["size"]
        )

        # Télécharger les images (gros fichier)
        download_file(
            FILES["images"]["url"],
            FILES["images"]["dest"],
            FILES["images"]["size"]
        )

        # Extraire les images
        img_dir = DATA_DIR / "ISIC_2019_Training_Input"
        if FILES["images"]["dest"].exists():
            extract_images(FILES["images"]["dest"], img_dir)

        # Vérification finale
        verify_dataset()

    except KeyboardInterrupt:
        print("\n\n⚠️  Téléchargement interrompu par l'utilisateur")
        print("   Relancez le script pour reprendre là où vous vous êtes arrêté")
    except Exception as e:
        print(f"\n\n❌ Erreur fatale : {e}")
        print("   Vérifiez votre connexion internet et réessayez")


if __name__ == "__main__":
    main()