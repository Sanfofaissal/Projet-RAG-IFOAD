# 🎓 Assistant RAG IFOAD-UJKZ

**Projet Data Science 2026 — Master 1 IFOAD · Université Joseph Ki-Zerbo**  
**Enseignant : Dr Delwende D. Arthur SAWADOGO**
**Groupe 2 : OUEDRAOGO Kiswendsida Marie Monique et SANFO Faïssal**
---
**Lien streamlit : https://projet-rag-ifoad-5pmwg7epkxsw8ntrcawkun.streamlit.app/**
## 📌 Présentation

Cet assistant intelligent permet aux étudiants et candidats de l'IFOAD-UJKZ d'obtenir des réponses précises sur :

- Les **formations disponibles** (Master Data Science, Licences, Certificats)
- Les **conditions de candidature** et pièces à fournir
- Les **frais d'inscription et de formation**
- La **maquette de cours** du Master  Sciences des Données
- Les **dates importantes** : examens, dépôts de projets, visioconférences

L'agent ne se contente pas de "parler" — il **analyse des documents officiels** pour guider précisément l'utilisateur, et sait dire *"je ne sais pas"* quand l'information n'est pas dans sa base de connaissances.

---

## 🏗️ Architecture du système

```
Question utilisateur
        │
        ▼
┌─────────────────────┐
│  Embeddings         │  paraphrase-multilingual-MiniLM-L12-v2
│  (Sentence-Trans.)  │  Gratuit · Multilingue · Local
└────────┬────────────┘
         │ vecteur de la question
         ▼
┌─────────────────────┐
│  Recherche          │  Top 4 chunks les plus proches
│  ChromaDB           │  Filtre similarité (seuil 0.55)
└────────┬────────────┘
         │ contexte pertinent
         ▼
┌─────────────────────┐
│  LLM Groq           │  Llama 3.3 70B · Gratuit · Ultra-rapide
│  (Génération)       │  Temperature 0.2 · Réponses factuelles
└────────┬────────────┘
         │ réponse
         ▼
┌─────────────────────┐
│  Interface          │  Streamlit · Chat interactif
│  Streamlit          │  Questions suggérées · Sources citées
└─────────────────────┘
```

---

## 📂 Sources de données

| Source | Type | Contenu |
|--------|------|---------|
| Communiqués officiels UJKZ | PDF (scannés) | Conditions de candidature, frais d'inscription, modalités |
| Calendrier Moodle | Fichier .ics | Examens, dépôts de projets, visioconférences |
| Maquette de cours | CSV + structuration manuelle | 22 cours S1+S2 du Master Data Science |

> **Note :** les PDF étant des scans, une étape OCR automatique (Tesseract, langue française) est appliquée avant la vectorisation.

---

## ⚙️ Stack technique

| Composant | Technologie | Justification |
|-----------|-------------|---------------|
| Extraction PDF | `pypdf` + `pytesseract` | Gestion des PDF textuels ET scannés |
| Chunking | `langchain-text-splitters` | Découpage sémantique avec overlap |
| Embeddings | `sentence-transformers` | Gratuit, multilingue, local |
| Base vectorielle | `ChromaDB` | Légère, persistante, sans serveur |
| LLM | Groq API (Llama 3.3 70B) | Gratuit, < 1s de latence |
| Interface | `Streamlit` | Standard Data Science, déploiement facile |

---

## 🗂️ Structure du projet

```
📁 projet-rag-ifoad/
  ├── app.py                                      # Application Streamlit
  ├── requirements.txt                            # Dépendances Python
  ├── chroma_db_ifoad/                            # Base vectorielle ChromaDB
  │     └── chroma.sqlite3
  ├── notebooks/
  │     ├── 01_ingestion_vectorisation_v4.ipynb   # Étape 1 : ingestion & vectorisation
  │     └── 02_agent_interface.ipynb              # Étape 2 : agent & interface
  └── README.md
```

---

## 🚀 Lancement local

### Prérequis
- Python 3.10+
- Une clé API Groq gratuite : [console.groq.com](https://console.groq.com)

### Installation

```bash
git clone https://github.com/votre-username/projet-rag-ifoad.git
cd projet-rag-ifoad
pip install -r requirements.txt
```

### Configuration

```bash
export GROQ_API_KEY="gsk_votre_clé_ici"
```

### Lancement

```bash
streamlit run app.py
```

---

## ☁️ Déploiement (Streamlit Community Cloud)

1. Forkez ce dépôt sur GitHub
2. Connectez-vous sur [share.streamlit.io](https://share.streamlit.io)
3. New app → sélectionnez ce repo → fichier : `app.py`
4. Dans **Advanced settings → Secrets**, ajoutez :
```toml
GROQ_API_KEY = "gsk_votre_clé_ici"
```
5. Deploy ✅

---

## 📓 Reproduction de la base vectorielle (Étape 1)

Si vous souhaitez reconstruire la base ChromaDB depuis les sources :

1. Ouvrez `notebooks/01_ingestion_vectorisation_v4.ipynb` dans Google Colab
2. Placez vos PDF dans Google Drive (`My Drive/contents/pdf/`)
3. Exécutez toutes les cellules (**Runtime → Run all**)
4. Téléchargez le dossier `chroma_db_ifoad/` généré

---

## 📊 Évaluation du système RAG

| Question test | Distance retrieval | Résultat |
|---------------|-------------------|----------|
| Frais d'inscription Master Data Science | 0.485 | ✅ Bon document remonté |
| Date limite projet final | 0.412 | ✅ Calendrier Moodle trouvé |
| Quels cours au programme Master ? | < 0.55 | ✅ Maquette de cours |
| Météo à Ouagadougou (hors-sujet) | 1.051 | ✅ Agent dit "je ne sais pas" |

> Le seuil de similarité est fixé à **0.55** : au-delà, l'agent refuse de répondre pour éviter les hallucinations.

---

