"""
Assistant RAG IFOAD-UJKZ — Application Streamlit
Étape 2 : Interface utilisateur + logique de l'agent

Stack :
- ChromaDB       : base vectorielle (construite dans le notebook Étape 1)
- Sentence-Transformers : mêmes embeddings que l'étape 1
- Groq API       : LLM Llama 3.3 70B (gratuit, ultra-rapide)
- Streamlit      : interface web de chat
"""

import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CHROMA_DB_PATH     = "./chroma_db_ifoad"   # chemin vers la base vectorielle (étape 1)
COLLECTION_NAME    = "ifoad_communiques"
EMBEDDING_MODEL    = "paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL         = "llama-3.3-70b-versatile"
N_RESULTS          = 5       # nombre de chunks récupérés par requête
SIMILARITY_CUTOFF  = 0.75    # seuil en dessous duquel on considère qu'on ne sait pas (distance cosinus)

SYSTEM_PROMPT = """Tu es l'Assistant Officiel de l'IFOAD-UJKZ (Institut de Formation Ouverte et à Distance de l'Université Joseph Ki-Zerbo), au Burkina Faso.

Ton rôle est de répondre avec précision aux questions des étudiants et candidats concernant :
- Les formations proposées (Master Data Science, Licences, Certificats…)
- Les conditions de candidature et pièces à fournir
- Les frais d'inscription et de formation
- Les calendriers académiques, dates d'examens et dépôts de projets
- Les modalités d'enseignement (en ligne, présentiel, plateforme Moodle…)

RÈGLES IMPORTANTES :
1. Tu réponds UNIQUEMENT à partir des informations fournies dans le CONTEXTE ci-dessous.
2. Si le contexte ne contient pas l'information demandée, réponds exactement : "Je ne dispose pas de cette information dans ma base de connaissances actuelle. Je vous invite à contacter directement l'IFOAD au 63375257 ou à écrire à urbain.traore@ujkz.bf"
3. Ne jamais inventer de chiffres, de dates ou de noms.
4. Cite toujours la source (nom du document et formation concernée) à la fin de ta réponse.
5. Réponds en français, avec un ton professionnel et bienveillant.

CONTEXTE :
{context}"""


# ─────────────────────────────────────────────
# INITIALISATION (mise en cache pour éviter
# de recharger à chaque interaction)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement de la base de connaissances…")
def load_resources():
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    return embedding_model, collection

@st.cache_resource(show_spinner=False)
def load_groq_client():
    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# ─────────────────────────────────────────────
# LOGIQUE RAG
# ─────────────────────────────────────────────
def retrieve_context(query: str, embedding_model, collection) -> tuple[str, list[dict]]:
    """Recherche les chunks les plus pertinents pour la question posée."""
    query_vec = embedding_model.encode([query], normalize_embeddings=True)
    results = collection.query(
        query_embeddings=query_vec.tolist(),
        n_results=N_RESULTS,
    )
    sources = []
    context_parts = []

    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        if distance > SIMILARITY_CUTOFF:
            continue  # chunk trop éloigné sémantiquement → ignoré
        doc_text = results["documents"][0][i]
        meta     = results["metadatas"][0][i]
        context_parts.append(doc_text)
        sources.append({
            "formation": meta.get("formation", "—"),
            "type_doc":  meta.get("type_doc", "—"),
            "fichier":   meta.get("source_file", "—"),
            "page":      meta.get("page", "—"),
            "distance":  round(distance, 3),
        })

    context = "\n\n---\n\n".join(context_parts) if context_parts else ""
    return context, sources


def ask_llm(question: str, context: str, history: list, groq_client) -> str:
    """Interroge le LLM Groq avec le contexte RAG et l'historique de conversation."""
    if not groq_client:
        return "⚠️ Clé API Groq manquante. Renseignez GROQ_API_KEY dans les secrets Streamlit."

    if not context:
        return (
            "Je ne dispose pas de cette information dans ma base de connaissances actuelle. "
            "Je vous invite à contacter directement l'IFOAD au **63375257** "
            "ou à écrire à **urbain.traore@ujkz.bf**."
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    # On inclut les 4 derniers échanges de l'historique pour le contexte conversationnel
    for msg in history[-8:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,   # faible pour des réponses factuelles et stables
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# INTERFACE STREAMLIT
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Assistant IFOAD-UJKZ",
        page_icon="🎓",
        layout="centered",
    )

    # ── CSS personnalisé ──────────────────────
    st.markdown("""
    <style>
        /* Palette : vert forêt UJKZ + blanc cassé + or */
        :root {
            --ujkz-green : #1B5E20;
            --ujkz-light : #E8F5E9;
            --ujkz-gold  : #F9A825;
            --text-main  : #1A1A1A;
            --text-muted : #555;
        }

        /* En-tête */
        .header-band {
            background: var(--ujkz-green);
            color: white;
            padding: 1.2rem 1.5rem 0.8rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
        }
        .header-band h1 { font-size: 1.4rem; margin: 0; font-weight: 700; }
        .header-band p  { margin: 0.2rem 0 0; font-size: 0.88rem; opacity: 0.85; }

        /* Bulles de chat */
        .chat-user {
            background: var(--ujkz-green);
            color: white;
            padding: 0.7rem 1rem;
            border-radius: 18px 18px 4px 18px;
            margin: 0.4rem 0 0.4rem 15%;
            font-size: 0.95rem;
        }
        .chat-assistant {
            background: var(--ujkz-light);
            color: var(--text-main);
            padding: 0.7rem 1rem;
            border-radius: 18px 18px 18px 4px;
            margin: 0.4rem 15% 0.4rem 0;
            font-size: 0.95rem;
            border-left: 3px solid var(--ujkz-green);
        }

        /* Sources */
        .source-tag {
            display: inline-block;
            background: #f0f0f0;
            color: var(--text-muted);
            font-size: 0.72rem;
            padding: 2px 8px;
            border-radius: 20px;
            margin: 2px 2px 0 0;
        }

        /* Disclaimer */
        .disclaimer {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-align: center;
            margin-top: 1rem;
            border-top: 1px solid #eee;
            padding-top: 0.5rem;
        }

        /* Cacher le menu Streamlit */
        #MainMenu, footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

    # ── En-tête ───────────────────────────────
    st.markdown("""
    <div class="header-band">
        <h1>🎓 Assistant IFOAD-UJKZ</h1>
        <p>Université Joseph Ki-Zerbo · Institut de Formation Ouverte et à Distance</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Chargement des ressources ─────────────
    try:
        embedding_model, collection = load_resources()
        groq_client = load_groq_client()
        n_docs = collection.count()
        st.caption(f"📚 Base de connaissances : **{n_docs} segments** indexés")
    except Exception as e:
        st.error(f"❌ Impossible de charger la base ChromaDB : {e}")
        st.info("Vérifiez que le dossier `chroma_db_ifoad/` est bien présent à côté de `app.py`.")
        return

    # ── Clé Groq (si non définie dans les secrets) ──
    if not groq_client:
        with st.sidebar:
            st.subheader("🔑 Clé API Groq")
            api_key_input = st.text_input(
                "Entrez votre clé Groq API",
                type="password",
                placeholder="gsk_...",
                help="Obtenez une clé gratuite sur console.groq.com"
            )
            if api_key_input:
                os.environ["GROQ_API_KEY"] = api_key_input
                st.cache_resource.clear()
                st.rerun()
            st.markdown("[Obtenir une clé gratuite →](https://console.groq.com)")

    # ── Historique de conversation ────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Message d'accueil
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "Bonjour ! Je suis l'assistant de l'IFOAD-UJKZ. "
                "Je peux vous renseigner sur les **formations disponibles**, "
                "les **conditions de candidature**, les **frais d'inscription**, "
                "et les **dates importantes** de l'agenda académique.\n\n"
                "Comment puis-je vous aider ?"
            ),
            "sources": [],
        })

    # ── Affichage des messages ────────────────
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-assistant">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                cols = st.columns([1])
                with cols[0]:
                    for src in msg["sources"]:
                        st.markdown(
                            f'<span class="source-tag">📄 {src["formation"]} — {src["type_doc"]}</span>',
                            unsafe_allow_html=True,
                        )

    # ── Zone de saisie ────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    question = st.chat_input("Posez votre question sur l'IFOAD…")

    if question:
        # Ajout de la question à l'historique
        st.session_state.messages.append({"role": "user", "content": question})
        st.markdown(f'<div class="chat-user">{question}</div>', unsafe_allow_html=True)

        # Récupération du contexte et génération de la réponse
        with st.spinner("Recherche en cours…"):
            context, sources = retrieve_context(question, embedding_model, collection)
            answer = ask_llm(
                question, context,
                [m for m in st.session_state.messages if m["role"] != "system"],
                groq_client,
            )

        # Affichage de la réponse
        st.markdown(f'<div class="chat-assistant">{answer}</div>', unsafe_allow_html=True)
        if sources:
            for src in sources:
                st.markdown(
                    f'<span class="source-tag">📄 {src["formation"]} — {src["type_doc"]}</span>',
                    unsafe_allow_html=True,
                )

        # Sauvegarde dans l'historique
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

    # ── Sidebar : questions suggérées ─────────
    with st.sidebar:
        st.subheader("💡 Questions fréquentes")
        suggestions = [
            "Quelles sont les conditions pour le Master Data Science ?",
            "Quels sont les frais d'inscription en Licence ?",
            "Quand est la date limite pour rendre le projet final ?",
            "Comment soumettre ma candidature sur Campus Faso ?",
            "Quelle est la durée de la formation en Certificat Numérique ?",
        ]
        for s in suggestions:
            if st.button(s, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": s})
                with st.spinner("Recherche en cours…"):
                    context, sources = retrieve_context(s, embedding_model, collection)
                    answer = ask_llm(
                        s, context,
                        [m for m in st.session_state.messages if m["role"] != "system"],
                        groq_client,
                    )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
                st.rerun()

        st.divider()
        if st.button("🗑️ Effacer la conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown(
        '<div class="disclaimer">Les réponses sont générées à partir des documents officiels de l\'IFOAD-UJKZ. '
        'En cas de doute, consultez directement l\'institution.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
