import streamlit as st
from openai import OpenAI # Eğer Google AI yerine OpenRouter kullanacaksanız bu kalmalı, aksi halde kaldırılabilir
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings # VEYA OpenRouter/Ollama karşılıkları
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import traceback
import uuid
import requests # YENİ
from bs4 import BeautifulSoup # YENİ
# import trafilatura # Eğer kullanacaksanız

# -----------------------------------------------------------------------------
# SAYFA KONFİGÜRASYONU
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Çok Kaynaklı AI Asistanı", page_icon="🌐")
# -----------------------------------------------------------------------------

# os.environ["TOKENIZERS_PARALLELISM"] = "false" # HuggingFace için

# --- API Konfigürasyonu (Google AI veya başka bir sağlayıcı) ---
# Örnek Google AI (secrets.toml dosyanızda ayarlı olmalı)
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
GOOGLE_LLM_MODEL_NAME = st.secrets.get("GOOGLE_LLM_MODEL_NAME", "gemini-1.5-flash-latest")
GOOGLE_EMBEDDING_MODEL_NAME = st.secrets.get("GOOGLE_EMBEDDING_MODEL_NAME", "models/embedding-001")

if not GOOGLE_API_KEY: # Veya kullandığınız diğer API için anahtar kontrolü
    st.error("API anahtarı bulunamadı! Lütfen Streamlit Secrets bölümünü kontrol edin.")
    st.stop()

try:
    llm_client = ChatGoogleGenerativeAI(model=GOOGLE_LLM_MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.1)
    embeddings_model_global = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL_NAME, google_api_key=GOOGLE_API_KEY)
    print("Google AI istemcileri başarıyla bağlandı.")
except Exception as e:
    st.error(f"AI istemcileri oluşturulurken hata: {e}"); st.stop()


# --- Web Sitesi İçeriği Çekme Fonksiyonu ---
def get_website_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # HTTP hataları için exception fırlat
        
        # Alternatif: trafilatura ile daha temiz metin çıkarma
        # downloaded = trafilatura.fetch_url(url)
        # if downloaded:
        #     main_text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
        #     if main_text:
        #         return main_text
        # return "" # Trafilatura başarısız olursa

        soup = BeautifulSoup(response.content, 'lxml') # 'lxml' daha hızlıdır
        
        # Gereksiz etiketleri kaldır (script, style, nav, footer vb.)
        for script_or_style in soup(["script", "style", "header", "footer", "nav", "aside", "form"]):
            script_or_style.decompose()
        
        # Sadece ana içerik alanını bulmaya çalışabiliriz (site yapısına göre değişir)
        # Örneğin: body, main, article etiketleri
        # Bu kısım sitenin yapısına göre özelleştirilebilir.
        # Şimdilik tüm görünür metni almayı deneyelim:
        text_parts = []
        # for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'span', 'div']):
        #     text_parts.append(element.get_text(separator=" ", strip=True))
        # text = " ".join(filter(None, text_parts))
        
        # Veya daha basitçe tüm body metnini alalım (daha gürültülü olabilir)
        body = soup.find('body')
        if body:
            text = body.get_text(separator='\n', strip=True) # Satır başlarını korumak için \n
            # Çoklu boşlukları ve satırları tek bir taneye indirge
            text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
            return text
        return ""
    except requests.exceptions.RequestException as e:
        st.error(f"Web sitesi içeriği çekilirken hata: {url} - {e}")
        return None
    except Exception as e:
        st.error(f"Web sitesi içeriği işlenirken beklenmedik hata: {url} - {e}")
        return None

# --- Diğer Yardımcı Fonksiyonlar (get_pdf_text, get_text_chunks, vb. öncekiyle aynı) ---
def get_pdf_text(pdf_docs):
    text = ""
    if pdf_docs:
        for pdf in pdf_docs:
            try:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    page_text = page.extract_text();
                    if page_text: text += page_text
            except Exception as e: st.warning(f"'{pdf.name}' dosyasından metin çıkarılırken hata: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250, length_function=len) # Web içeriği için chunk size'ı biraz artırabiliriz
    return text_splitter.split_text(text)

def create_vector_store_from_chunks(text_chunks, current_embeddings_model):
    if not text_chunks or not current_embeddings_model: return None
    try: return FAISS.from_texts(texts=text_chunks, embedding=current_embeddings_model)
    except Exception as e: st.error(f"Vektör deposu oluşturulurken hata: {e}"); st.error(traceback.format_exc()); return None

def get_conversational_chain_prompt_template(): # Bu aynı kalabilir
    prompt_template_str = """
    SENİN GÖREVİN: Sadece ve sadece aşağıda "Bağlam:" olarak verilen metindeki bilgileri kullanarak "Soru:" kısmındaki soruyu yanıtlamaktır.
    KESİNLİKLE DIŞARIDAN BİLGİ KULLANMA, YORUM YAPMA, EK AÇIKLAMA EKLEME VEYA CEVAP UYDURMA.
    Cevabın SADECE ve SADECE "Bağlam:" içindeki bilgilere dayanmalıdır.

    Eğer "Soru:" kısmındaki soruya cevap "Bağlam:" içinde bulunmuyorsa, şu cevabı ver:
    "Bu bilgi sağlanan kaynakta (PDF/Web Sitesi) bulunmuyor."
    BU CEVABIN DIŞINDA HİÇBİR ŞEY EKLEME. Örneğin, "Bu bilgi kaynakta yok ama genel olarak şöyledir..." GİBİ BİR AÇIKLAMA YAPMA.

    Bağlam:
    {context}

    Soru: {question}

    Cevap:"""
    return PromptTemplate(template=prompt_template_str, input_variables=["context", "question"])

# --- Session State ve Oturum Yönetimi (Genişletilmiş) ---
if "sessions" not in st.session_state: st.session_state.sessions = {}
if "current_session_id" not in st.session_state: st.session_state.current_session_id = None
if "prompt_template" not in st.session_state: st.session_state.prompt_template = get_conversational_chain_prompt_template()

def create_new_session(session_type="pdf"): # session_type: "pdf" veya "website"
    session_id = str(uuid.uuid4())
    type_prefix = "PDF Sohbeti" if session_type == "pdf" else "Web Sohbeti"
    session_name = f"{type_prefix} {len(st.session_state.sessions) + 1}"
    st.session_state.sessions[session_id] = {
        "id": session_id, "name": session_name,
        "source_type": session_type, # Kaynak türünü sakla
        "source_info": None, # PDF adları listesi veya web sitesi URL'si
        "vector_store": None, "chat_history": [], "processed": False
    }
    st.session_state.current_session_id = session_id
    return session_id

# ... (get_active_session_data, delete_session fonksiyonları aynı kalabilir) ...
def get_active_session_data():
    if st.session_state.current_session_id and st.session_state.current_session_id in st.session_state.sessions:
        return st.session_state.sessions[st.session_state.current_session_id]
    return None
def delete_session(session_id_to_delete):
    if session_id_to_delete in st.session_state.sessions:
        del st.session_state.sessions[session_id_to_delete]
        if st.session_state.current_session_id == session_id_to_delete:
            st.session_state.current_session_id = None
            if st.session_state.sessions: st.session_state.current_session_id = list(st.session_state.sessions.keys())[0]

st.title("🌐 Çok Kaynaklı AI Asistanı")

# --- Kenar Çubuğu ---
with st.sidebar:
    st.header("Sohbet Yönetimi")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Yeni PDF Sohbeti", key="new_pdf_chat"):
            create_new_session(session_type="pdf"); st.rerun()
    with col2:
        if st.button("🔗 Yeni Web Sohbeti", key="new_web_chat"):
            create_new_session(session_type="website"); st.rerun()

    session_options = {sid: f"{sdata['name']} ({sdata.get('source_info', 'Kaynak Yok') if isinstance(sdata.get('source_info'), str) else ', '.join(sdata.get('source_info', ['Kaynak Yok'])) if sdata.get('source_info') else 'Kaynak Yok'})"
                       for sid, sdata in st.session_state.sessions.items()}

    if not session_options and st.session_state.current_session_id is None:
        create_new_session(); st.rerun() # Varsayılan olarak PDF sohbeti başlat

    if session_options:
        selected_session_id = st.selectbox(
            "Aktif Sohbeti Seçin:", options=list(session_options.keys()),
            format_func=lambda sid: session_options[sid],
            index=list(session_options.keys()).index(st.session_state.current_session_id) if st.session_state.current_session_id in session_options else 0,
            key="session_selector"
        )
        if selected_session_id != st.session_state.current_session_id:
            st.session_state.current_session_id = selected_session_id; st.rerun()

        active_session = get_active_session_data()
        if active_session:
            st.markdown("---"); st.subheader(f"Aktif: {active_session['name']}")

            raw_text = None
            if active_session["source_type"] == "pdf":
                uploader_key = f"pdf_uploader_{active_session['id']}"
                uploaded_files = st.file_uploader("PDF dosyalarını yükleyin:", accept_multiple_files=True, type="pdf", key=uploader_key)
                if st.button("PDF'leri İşle", key=f"process_pdf_{active_session['id']}"):
                    if uploaded_files:
                        active_session["source_info"] = [f.name for f in uploaded_files]
                        raw_text = get_pdf_text(uploaded_files)
                    else: st.warning("Lütfen PDF dosyası yükleyin.")
            
            elif active_session["source_type"] == "website":
                url_input_key = f"url_input_{active_session['id']}"
                website_url = st.text_input("Web sitesi URL'sini girin:", key=url_input_key, placeholder="https://ornek.com/sayfa")
                if st.button("Web Sitesini İşle", key=f"process_web_{active_session['id']}"):
                    if website_url and website_url.startswith(("http://", "https://")):
                        active_session["source_info"] = website_url
                        with st.spinner(f"{website_url} içeriği çekiliyor ve işleniyor..."):
                            raw_text = get_website_text(website_url)
                    elif website_url: st.warning("Lütfen geçerli bir URL girin (http:// veya https:// ile başlayan).")
                    else: st.warning("Lütfen bir URL girin.")

            # Ortak İşleme Mantığı (raw_text elde edildiyse)
            if raw_text is not None: # Bu, bir işlem butonuna basıldığı ve kaynak okunduğu anlamına gelir
                with st.spinner("İçerik işleniyor..."):
                    if not raw_text.strip():
                        st.error("Kaynak boş veya metin çıkarılamadı.")
                        active_session["vector_store"] = None; active_session["processed"] = False
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        if not text_chunks:
                            st.error("Metin parçalara ayrılamadı.")
                            active_session["vector_store"] = None; active_session["processed"] = False
                        else:
                            vector_store = create_vector_store_from_chunks(text_chunks, embeddings_model_global)
                            if vector_store:
                                active_session["vector_store"] = vector_store
                                active_session["chat_history"] = [] # Yeni kaynak işlenince sohbeti sıfırla
                                active_session["processed"] = True
                                st.success(f"Kaynak '{active_session['name']}' için başarıyla işlendi.")
                                st.rerun()
                            else:
                                st.error("Vektör deposu oluşturulamadı."); active_session["processed"] = False
            
            if active_session["processed"]:
                 source_display = active_session["source_info"]
                 if isinstance(source_display, list): source_display = ", ".join(source_display)
                 st.markdown(f"**İşlenmiş Kaynak:** {source_display}")

            st.markdown("---")
            if st.button(f"'{active_session['name']}' Oturumunu Sil", type="secondary", key=f"delete_btn_{active_session['id']}"):
                delete_session(active_session['id']); st.success(f"Oturum silindi."); st.rerun()
    else:
        st.sidebar.info("Henüz bir sohbet oturumu yok.")


# --- Ana Sohbet Alanı (öncekiyle büyük ölçüde aynı) ---
active_session_data = get_active_session_data()
if active_session_data:
    st.header(f"Sohbet: {active_session_data['name']}")
    if active_session_data.get("source_info"):
        source_display = active_session_data["source_info"]
        if isinstance(source_display, list): source_display = ", ".join(source_display)
        st.caption(f"Kaynak: {source_display}")

    for message in active_session_data["chat_history"]:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if user_query := st.chat_input(f"Kaynak hakkında sorun..."):
        if not active_session_data.get("vector_store"):
            st.warning("Bu sohbet için henüz bir kaynak (PDF/Web Sitesi) işlenmedi veya vektör deposu oluşturulamadı.")
        else:
            # ... (LLM çağrı mantığı öncekiyle aynı kalır) ...
            active_session_data["chat_history"].append({"role": "user", "content": user_query})
            with st.chat_message("user"): st.markdown(user_query)
            with st.chat_message("assistant"):
                message_placeholder = st.empty(); full_response_text = ""
                try:
                    docs = active_session_data["vector_store"].similarity_search(query=user_query, k=5) # Daha fazla chunk alabiliriz
                    if not docs:
                        full_response_text = "Bu bilgi sağlanan kaynakta (PDF/Web Sitesi) bulunmuyor."
                    else:
                        context_text = "\n\n".join([doc.page_content for doc in docs])
                        current_prompt_template = st.session_state.prompt_template
                        formatted_prompt = current_prompt_template.format(context=context_text, question=user_query)
                        
                        for chunk in llm_client.stream(formatted_prompt): # veya messages_for_llm yapısı
                            if hasattr(chunk, 'content'):
                                full_response_text += chunk.content
                                message_placeholder.markdown(full_response_text + "▌")
                    message_placeholder.markdown(full_response_text)
                except Exception as e:
                    st.error(f"Yanıt alınırken bir hata oluştu: {e}"); st.error(traceback.format_exc())
                    full_response_text = "Üzgünüm, bir hata oluştu."; message_placeholder.markdown(full_response_text)
            active_session_data["chat_history"].append({"role": "assistant", "content": full_response_text})
else:
    st.info("Lütfen bir sohbet seçin veya yeni bir tane başlatın.")

st.sidebar.markdown("---")
st.sidebar.caption(f"LLM: {GOOGLE_LLM_MODEL_NAME}")
st.sidebar.caption(f"Embedding: {GOOGLE_EMBEDDING_MODEL_NAME}")
