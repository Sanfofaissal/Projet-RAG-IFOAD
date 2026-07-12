
import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

CHROMA_DB_PATH    = "/content/chroma_db_ifoad"
COLLECTION_NAME   = "ifoad_communiques"
EMBEDDING_MODEL   = "paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL        = "llama-3.3-70b-versatile"
N_RESULTS         = 4
SIMILARITY_CUTOFF = 0.55

SYSTEM_PROMPT = """Tu es l'Assistant Officiel de l'IFOAD-UJKZ (Institut de Formation Ouverte et à Distance de l'Université Joseph Ki-Zerbo), au Burkina Faso.

Ton rôle est de répondre avec précision aux questions des étudiants et candidats concernant :
- Les formations proposées (Master Data Science, Licences, Certificats...)
- Les conditions de candidature et pièces à fournir
- Les frais d'inscription et de formation
- Les calendriers académiques, dates d'examens et dépôts de projets
- Les modalités d'enseignement (en ligne, présentiel, plateforme Moodle...)

RÈGLES IMPORTANTES :
1. Réponds UNIQUEMENT à partir des informations fournies dans le CONTEXTE ci-dessous.
2. Si le contexte ne contient pas l'information demandée, réponds exactement : "Je ne dispose pas de cette information dans ma base de connaissances actuelle. Je vous invite à contacter directement l'IFOAD au 63375257 ou à écrire à urbain.traore@ujkz.bf"
3. Ne jamais inventer de chiffres, de dates ou de noms.
4. Cite toujours la source (nom du document et formation concernée) à la fin de ta réponse.
5. Réponds en français, avec un ton professionnel et bienveillant.

CONTEXTE :
{context}"""

@st.cache_resource(show_spinner="Chargement de la base de connaissances...")
def load_resources():
    emb = SentenceTransformer(EMBEDDING_MODEL)
    cli = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    col = cli.get_collection(COLLECTION_NAME)
    return emb, col

@st.cache_resource(show_spinner=False)
def load_groq():
    key = os.environ.get("GROQ_API_KEY", "")
    return Groq(api_key=key) if key else None

def retrieve(query, emb, col):
    vec = emb.encode([query], normalize_embeddings=True)
    res = col.query(query_embeddings=vec.tolist(), n_results=N_RESULTS)
    sources, parts = [], []
    for i in range(len(res["ids"][0])):
        if res["distances"][0][i] > SIMILARITY_CUTOFF:
            continue
        parts.append(res["documents"][0][i])
        m = res["metadatas"][0][i]
        sources.append({"formation": m.get("formation","-"), "type_doc": m.get("type_doc","-")})
    return "\n\n---\n\n".join(parts), sources

def answer(question, context, history, groq_cli):
    if not groq_cli:
        return "⚠️ Clé API Groq manquante."
    if not context:
        return "Je ne dispose pas de cette information dans ma base de connaissances actuelle. Contactez l'IFOAD au **63375257** ou **urbain.traore@ujkz.bf**."
    msgs = [{"role":"system","content":SYSTEM_PROMPT.format(context=context)}]
    for m in history[-8:]:
        msgs.append({"role":m["role"],"content":m["content"]})
    msgs.append({"role":"user","content":question})
    r = groq_cli.chat.completions.create(model=GROQ_MODEL, messages=msgs, temperature=0.2, max_tokens=1024)
    return r.choices[0].message.content

def main():
    st.set_page_config(page_title="Assistant IFOAD-UJKZ", page_icon="🎓", layout="centered")
    st.markdown('''<style>
    .header{background:#1B5E20;color:white;padding:1.2rem 1.5rem 0.8rem;border-radius:12px;margin-bottom:1.5rem}
    .header h1{font-size:1.4rem;margin:0;font-weight:700}
    .header p{margin:.2rem 0 0;font-size:.88rem;opacity:.85}
    .cu{background:#1B5E20;color:white;padding:.7rem 1rem;border-radius:18px 18px 4px 18px;margin:.4rem 0 .4rem 15%;font-size:.95rem}
    .ca{background:#E8F5E9;padding:.7rem 1rem;border-radius:18px 18px 18px 4px;margin:.4rem 15% .4rem 0;font-size:.95rem;border-left:3px solid #1B5E20}
    .st{display:inline-block;background:#f0f0f0;color:#555;font-size:.72rem;padding:2px 8px;border-radius:20px;margin:2px 2px 0 0}
    #MainMenu,footer{visibility:hidden}
    </style>''', unsafe_allow_html=True)
    st.markdown('<div class="header"><h1>🎓 Assistant IFOAD-UJKZ</h1><p>Université Joseph Ki-Zerbo · Institut de Formation Ouverte et à Distance</p></div>', unsafe_allow_html=True)

    try:
        emb, col = load_resources()
        groq_cli = load_groq()
        st.caption(f"📚 Base de connaissances : **{col.count()} segments** indexés")
    except Exception as e:
        st.error(f"❌ Impossible de charger la base ChromaDB : {e}")
        return

    if not groq_cli:
        with st.sidebar:
            st.subheader("🔑 Clé API Groq")
            k = st.text_input("Clé Groq", type="password", placeholder="gsk_...")
            if k:
                os.environ["GROQ_API_KEY"] = k
                st.cache_resource.clear()
                st.rerun()
            st.markdown("[Obtenir une clé gratuite →](https://console.groq.com)")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role":"assistant","content":"Bonjour ! Je suis l'assistant de l'IFOAD-UJKZ. Je peux vous renseigner sur les **formations**, les **conditions de candidature**, les **frais d'inscription** et les **dates importantes**.\n\nComment puis-je vous aider ?","sources":[]}]

    for msg in st.session_state.messages:
        css = "cu" if msg["role"]=="user" else "ca"
        st.markdown(f'<div class="{css}">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            for s in msg["sources"]:
                st.markdown(f'<span class="st">📄 {s["formation"]} — {s["type_doc"]}</span>', unsafe_allow_html=True)

    q = st.chat_input("Posez votre question sur l'IFOAD...")
    if q:
        st.session_state.messages.append({"role":"user","content":q})
        st.markdown(f'<div class="cu">{q}</div>', unsafe_allow_html=True)
        with st.spinner("Recherche en cours..."):
            ctx, srcs = retrieve(q, emb, col)
            rep = answer(q, ctx, [m for m in st.session_state.messages if m["role"]!="system"], groq_cli)
        st.markdown(f'<div class="ca">{rep}</div>', unsafe_allow_html=True)
        for s in srcs:
            st.markdown(f'<span class="st">📄 {s["formation"]} — {s["type_doc"]}</span>', unsafe_allow_html=True)
        st.session_state.messages.append({"role":"assistant","content":rep,"sources":srcs})

    with st.sidebar:
        st.subheader("💡 Questions fréquentes")
        for s in ["Frais d'inscription Master Data Science ?","Conditions de candidature Licence ?","Date limite dépôt projet final ?","Comment candidater sur Campus Faso ?"]:
            if st.button(s, use_container_width=True):
                st.session_state.messages.append({"role":"user","content":s})
                ctx, srcs = retrieve(s, emb, col)
                rep = answer(s, ctx, st.session_state.messages, groq_cli)
                st.session_state.messages.append({"role":"assistant","content":rep,"sources":srcs})
                st.rerun()
        st.divider()
        if st.button("🗑️ Effacer la conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
