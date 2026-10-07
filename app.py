"""GANisme — Streamlit application: generate paintings with the GANs trained in this project.

Run:
    streamlit run app.py

The application only depends on the `models/` folder (generator weights and the `models.json`
description, both produced by `src/export_models.py`) and on the code in `src/`. It works without a
GPU: an image is generated in a few milliseconds on a CPU.

The interface is in French (the language of the project's audience); the code is in English.
"""
from __future__ import annotations

import io
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import generate as gn  # noqa: E402

MODELS_DIR = ROOT / "models"
DISPLAY_WIDTH = 256        # display width of a painting, in pixels
MAX_SEED = 999_999
GENRE_LABELS = {"portrait": "Portraits", "landscape": "Paysages"}      # interface labels

st.set_page_config(page_title="GANisme — générateur de peintures", page_icon="🎨", layout="wide")


# --------------------------------------------------------------------------- models
@st.cache_data
def load_cards() -> dict:
    """Description of each model (resolution, scores, hyperparameters)."""
    return json.loads((MODELS_DIR / "models.json").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Chargement du modèle…")
def load_model(key: str):
    """The generator is loaded once, then kept in memory across interactions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator, _ = gn.load_generator_file(MODELS_DIR / f"{key}.pt", device)
    return generator


@st.cache_data(show_spinner=False)
def paint(key: str, n: int, seed: int) -> np.ndarray:
    """n paintings (uint8, n × H × W × 3). Cached: the same seed gives the same images again."""
    return gn.sample(load_model(key), n, seed=seed)


@st.cache_data(show_spinner=False)
def morph(key: str, seed_a: int, seed_b: int, steps: int) -> np.ndarray:
    return gn.interpolate(load_model(key), seed_a, seed_b, steps)


# --------------------------------------------------------------------------- image helpers
def upscale(img: np.ndarray, width: int = DISPLAY_WIDTH) -> Image.Image:
    """Enlarge a painting for display (Lanczos: smooth enlargement, no blocky pixels)."""
    im = Image.fromarray(img)
    return im.resize((width, round(width * im.height / im.width)), Image.LANCZOS)


def png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def zip_bytes(images: np.ndarray, prefix: str) -> bytes:
    """Archive of the paintings at their original size, one per file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, img in enumerate(images, 1):
            zf.writestr(f"{prefix}_{i:02d}.png", png_bytes(Image.fromarray(img)))
    return buf.getvalue()


def gif_bytes(images: np.ndarray, width: int = DISPLAY_WIDTH, ms: int = 140) -> bytes:
    """Back-and-forth animation of an interpolation."""
    frames = [upscale(img, width) for img in images]
    frames = frames + frames[-2:0:-1]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=ms, loop=0)
    return buf.getvalue()


def new_seed() -> None:
    st.session_state.seed = random.randint(0, MAX_SEED)


# --------------------------------------------------------------------------- page
if not (MODELS_DIR / "models.json").exists():
    st.error("Aucun modèle trouvé dans `models/`. Lancez d'abord : `python src/export_models.py`.")
    st.stop()

cards = load_cards()
genres = {GENRE_LABELS[c["genre"]]: c["genre"] for c in cards.values()}          # "Portraits" -> "portrait"
if "seed" not in st.session_state:
    st.session_state.seed = 2024

with st.sidebar:
    st.header("Réglages")
    genre_label = st.radio("Genre", list(genres), horizontal=True)
    sizes = sorted((c["width"] for c in cards.values() if c["genre"] == genres[genre_label]), reverse=True)
    width = st.radio("Résolution", sizes, horizontal=True,
                     format_func=lambda w: f"{w} px de large" + (" (détaillée)" if w == max(sizes) else " (rapide)"))
    key = f"{genres[genre_label]}_{width}"
    card = cards[key]
    n = st.slider("Nombre de peintures", 4, 32, 12, step=4)
    st.number_input("Graine", 0, MAX_SEED, key="seed", step=1,
                    help="Le point de départ aléatoire. La même graine redonne exactement les mêmes peintures.")
    st.button("🎲 Nouvelles peintures", on_click=new_seed, type="primary", width="stretch")
    st.caption(f"Modèle : `{key}` · {card['width']}×{card['height']} px · "
               f"{card['params_G'] / 1e6:.1f} millions de paramètres")

seed = int(st.session_state.seed)
st.title("GANisme")
st.markdown(
    "Des **peintures qui n'existent pas**, produites par un réseau antagoniste génératif (GAN) entraîné sur "
    f"{card['n_train']:,} {genre_label.lower()} de la Web Gallery of Art (XVᵉ–XIXᵉ siècles). "
    "Le modèle ne reçoit aucune description : il part d'un simple tirage aléatoire.".replace(",", " "))

tab_generate, tab_morph, tab_about = st.tabs(["Générer", "Métamorphose", "À propos des modèles"])

# ---- Tab 1: generate
with tab_generate:
    images = paint(key, n, seed)
    per_row = 6 if card["height"] > card["width"] else 4
    for start in range(0, n, per_row):
        for col, i in zip(st.columns(per_row), range(start, min(start + per_row, n))):
            col.image(upscale(images[i]), caption=f"n° {i + 1}", width="stretch")

    left, right = st.columns(2)
    left.download_button("Télécharger les peintures (ZIP)", zip_bytes(images, f"{key}_graine{seed}"),
                         file_name=f"ganisme_{key}_graine{seed}.zip", mime="application/zip", width="stretch")
    right.download_button("Télécharger la planche (PNG)", png_bytes(gn.to_grid(images, ncol=per_row, scale=2)),
                          file_name=f"ganisme_{key}_graine{seed}.png", mime="image/png", width="stretch")
    st.caption(f"Graine {seed} · images générées en {card['width']}×{card['height']} px, agrandies à l'affichage. "
               "Les visages sont souvent déformés : c'est la principale limite de ces modèles.")

# ---- Tab 2: interpolation between two paintings
with tab_morph:
    st.markdown(
        "Chaque peinture correspond à un point d'un **espace latent** de 100 dimensions. En se déplaçant "
        "progressivement d'un point à un autre, on obtient une transformation continue : le signe que le modèle "
        "a appris un espace de peintures possibles, et non une collection d'images.")
    c1, c2, c3 = st.columns(3)
    seed_a = c1.number_input("Graine de départ", 0, MAX_SEED, seed, step=1)
    seed_b = c2.number_input("Graine d'arrivée", 0, MAX_SEED, (seed + 1) % (MAX_SEED + 1), step=1)
    steps = c3.slider("Nombre d'étapes", 6, 16, 8)
    frames = morph(key, int(seed_a), int(seed_b), steps)

    shown = frames[np.linspace(0, steps - 1, min(steps, 8)).round().astype(int)]
    for col, img in zip(st.columns(len(shown)), shown):
        col.image(upscale(img), width="stretch")

    animation = gif_bytes(frames)
    left, right = st.columns([1, 3])
    left.image(animation, caption="Animation aller-retour")
    right.download_button("Télécharger l'animation (GIF)", animation,
                          file_name=f"ganisme_{key}_{seed_a}_vers_{seed_b}.gif", mime="image/gif")

# ---- Tab 3: model cards
with tab_about:
    st.subheader("Les quatre modèles")
    rows = [{
        "Modèle": k, "Genre": GENRE_LABELS[c["genre"]], "Taille": f"{c['width']}×{c['height']}",
        "Images d'entraînement": c["n_train"], "Paramètres (M)": round(c["params_G"] / 1e6, 1),
        "Époques": c["epoch"], "Calcul (min)": c["train_minutes"], "FID": round(c["scores"]["FID"], 1),
        "Précision": round(c["scores"]["precision"], 2), "Rappel": round(c["scores"]["recall"], 2),
        "Copies (%)": round(100 * c["scores"]["copies"], 1),
    } for k, c in cards.items()]
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("Chaque genre et chaque résolution a sa propre échelle de référence : les FID de deux lignes "
               "différentes ne se comparent pas entre eux.")

    a, b = st.columns(2)
    with a:
        st.subheader("Comment lire ces chiffres")
        st.markdown(
            "- **FID** : distance entre les images générées et les vraies peintures. Plus il est bas, mieux c'est.\n"
            "- **Précision** : part des images générées qui ressemblent à de vraies peintures (qualité).\n"
            "- **Rappel** : part de la variété réelle que le modèle sait produire (diversité).\n"
            "- **Copies** : part des images générées plus proches du jeu d'entraînement que ne l'est une vraie "
            "peinture inconnue. Environ 1 % est attendu ; au-delà, le modèle recopierait ses données.")
    with b:
        st.subheader("Ce que font ces modèles")
        h = card["hyperparams"]
        st.markdown(
            "Un **DCGAN** : un générateur convolutif transforme 100 nombres aléatoires en image, pendant qu'un "
            "discriminateur apprend à distinguer ses productions des vraies peintures. Trois ingrédients "
            "équilibrent ce jeu :\n"
            f"- une **augmentation différentiable** des images vues par le discriminateur (`{h['diffaug']}`) ;\n"
            f"- la **normalisation spectrale** du discriminateur, et {h['n_dis']} mises à jour du discriminateur "
            "pour 1 du générateur ;\n"
            "- des hyperparamètres trouvés par une étude **Optuna** de 30 essais.")

    st.subheader("Limites")
    st.markdown(
        "- Les **visages** sont souvent déformés, surtout en basse résolution.\n"
        "- La **diversité** reste inférieure à celle des vraies peintures (rappel faible).\n"
        "- Le corpus est **biaisé** : peinture européenne du XVᵉ au XIXᵉ siècle, écoles italienne, française, "
        "hollandaise et flamande dominantes. Les images générées héritent de ce biais.\n"
        "- Hormis `portrait_64`, retenu parmi trois entraînements, chaque modèle n'a été entraîné qu'une fois : "
        "ses scores peuvent devoir une part au hasard.")
    st.caption("Projet GANisme — La Plateforme_. Démarche, analyses et résultats : dossier `notebooks/` du dépôt.")
