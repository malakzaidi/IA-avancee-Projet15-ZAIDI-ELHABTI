# diagnostic_chemins.py
from pathlib import Path
import os

print("🔍 RECHERCHE DU FICHIER H5 DANS VOTRE PROJET")
print("=" * 60)

# Chemin de votre projet
projet = Path(r"/")
print(f"📁 Projet: {projet}")

# Chercher tous les fichiers .h5 et .keras
print("\n📄 Fichiers .h5 trouvés:")
h5_files = list(projet.glob("**/*.h5"))
for f in h5_files:
    print(f"  ✅ {f.relative_to(projet)} ({f.stat().st_size / (1024*1024):.2f} MB)")

print("\n📄 Fichiers .keras trouvés:")
keras_files = list(projet.glob("**/*.keras"))
for f in keras_files:
    print(f"  ✅ {f.relative_to(projet)} ({f.stat().st_size / (1024*1024):.2f} MB)")

print("\n📁 Structure du dossier models/ (si existe):")
models_dir = projet / "models"
if models_dir.exists():
    for item in models_dir.iterdir():
        print(f"  📄 {item.name} ({'dossier' if item.is_dir() else f'{item.stat().st_size / 1024:.1f} KB'})")
else:
    print("  ❌ Dossier models/ n'existe pas")

# Vérifier si le fichier existe aux emplacements recherchés
print("\n🔎 Vérification des chemins recherchés par le code:")
search_paths = [
    projet / "models" / "skin_lesion_model.h5",
    projet / "skin_lesion_model.h5",
    projet / "backend" / "skin_lesion_model.h5",
]

for path in search_paths:
    exists = "✅ EXISTE" if path.exists() else "❌ N'EXISTE PAS"
    print(f"  {exists}: {path}")