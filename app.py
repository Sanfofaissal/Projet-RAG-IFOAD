import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
CHROMA_DB_PATH      = "./chroma_db_ifoad"
COLLECTION_NAME     = "ifoad_communiques"
EMBEDDING_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL          = "llama-3.3-70b-versatile"
N_RESULTS           = 4
SIMILARITY_CUTOFF   = 0.45   # seuil sur la SIMILARITÉ (1 - distance cosinus), pas la distance brute

SYSTEM_PROMPT = """Tu es l'Assistant Officiel de l'IFOAD-UJKZ (Institut de Formation Ouverte et à Distance, Université Joseph Ki-Zerbo).

Ton rôle est de répondre avec précision aux questions des étudiants et candidats sur :
- Les formations proposées (Master Data Science, Licences, Certificats...)
- Les conditions de candidature et pièces à fournir
- Les frais d'inscription et de formation
- Les calendriers académiques, dates d'examens et dépôts de projets

Règles strictes :
- Réponds UNIQUEMENT à partir du contexte fourni ci-dessous.
- Si l'information n'est pas dans le contexte, dis clairement que tu ne disposes pas de cette information et invite à contacter l'IFOAD (Tél. 63375257, urbain.traore@ujkz.bf).
- Ne jamais inventer de chiffres, dates ou conditions.
- Cite la source (référence du communiqué) quand c'est pertinent.
- Réponds en français, de façon claire et structurée."""


# ─────────────────────────────────────────
# CHARGEMENT DES RESSOURCES (mis en cache)
# ─────────────────────────────────────────
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception as e:
        st.error(f"❌ Impossible de charger la base ChromaDB : {e}")
        st.info(f"Vérifiez que le dossier {CHROMA_DB_PATH} est bien présent à côté de app.py.")
        st.stop()


@st.cache_resource
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("❌ Clé GROQ_API_KEY manquante dans les secrets Streamlit.")
        st.stop()
    return Groq(api_key=api_key)


embedding_model = load_embedding_model()
collection      = load_chroma_collection()
groq_client     = load_groq_client()


# ─────────────────────────────────────────
# RECHERCHE SÉMANTIQUE
# ─────────────────────────────────────────
def retrieve_context(query: str, n_results: int = N_RESULTS):
    query_vector = embedding_model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=n_results)

    chunks = []
    for i in range(len(results["ids"][0])):
        distance   = results["distances"][0][i]
        similarity = 1 - distance  # conversion distance cosinus → similarité
        if similarity >= SIMILARITY_CUTOFF:
            chunks.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": similarity,
            })
    return chunks


def build_prompt(query: str, chunks: list):
    if not chunks:
        return None

    context = "\n\n---\n\n".join(
        f"{c['text']}\n(Source : {c['metadata'].get('reference', 'N/A')})"
        for c in chunks
    )
    return f"""Contexte disponible :

{context}

---

Question de l'utilisateur : {query}

Réponds à la question en te basant uniquement sur le contexte ci-dessus."""


def ask_llm(query: str, chunks: list) -> str:
    prompt = build_prompt(query, chunks)

    if prompt is None:
        return (
            "Je ne dispose pas de cette information dans ma base de connaissances actuelle. "
            "Je vous invite à contacter directement l'IFOAD au 63375257 "
            "ou à écrire à urbain.traore@ujkz.bf."
        )
	response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────
# INTERFACE STREAMLIT
# ─────────────────────────────────────────
st.set_page_config(page_title="Assistant IFOAD-UJKZ", page_icon="🎓", layout="centered")

st.title("🎓 Assistant IFOAD-UJKZ")
st.caption("Université Joseph Ki-Zerbo · Institut de Formation Ouverte et à Distance")

try:
    st.info(f"📚 Base de connaissances : {collection.count()} segments indexés")
except Exception:
    pass

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Bonjour ! Je suis l'assistant de l'IFOAD-UJKZ. Je peux vous renseigner sur les "
                "formations disponibles, les conditions de candidature, les frais d'inscription, "
                "et les dates importantes de l'agenda académique.\n\nComment puis-je vous aider ?"
            ),
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Posez votre question sur l'IFOAD"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            chunks = retrieve_context(user_query)
            answer = ask_llm(user_query, chunks)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
