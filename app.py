# --- START OF FILE app2.py ---

import streamlit as st
# from openai import OpenAI # Eğer farklı bir LLM sağlayıcı kullanacaksanız
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings # VEYA kullandığınız diğer LLM/Embedding kütüphaneleri
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
# from langchain_core.messages import HumanMessage, SystemMessage # Eğer mesaj objeleriyle çalışacaksanız
import traceback
import uuid
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse # YENİ: URL'den domain almak için
# from duckduckgo_search import DDGS # Eğer 'ara:' özelliğini de tutacaksanız

# -----------------------------------------------------------------------------
# SAYFA KONFİGÜRASYONU
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Çok Kaynaklı AI Asistanı", page_icon="🌐")
# -----------------------------------------------------------------------------

# os.environ["TOKENIZERS_PARALLELISM"] = "false" # HuggingFace yerel modeller için

# --- API Konfigürasyonu (Örnek: Google AI) ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
GOOGLE_LLM_MODEL_NAME = st.secrets.get("GOOGLE_LLM_MODEL_NAME", "gemini-1.5-flash-latest")
GOOGLE_EMBEDDING_MODEL_NAME = st.secrets.get("GOOGLE_EMBEDDING_MODEL_NAME", "models/embedding-001")

if not GOOGLE_API_KEY:
    st.error("Google API anahtarı bulunamadı! Lütfen Streamlit Secrets bölümünü kontrol edin.")
    st.stop()

try:
    llm_client = ChatGoogleGenerativeAI(
        model=GOOGLE_LLM_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.15,
    )
    embeddings_model_global = GoogleGenerativeAIEmbeddings(
        model=GOOGLE_EMBEDDING_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY
    )
    print("Google AI istemcileri başarıyla bağlandı.")
except Exception as e:
    st.error(f"AI istemcileri oluşturulurken hata: {e}")
    st.error(traceback.format_exc())
    st.stop()

# --- Web Sitesi İçeriği Çekme Fonksiyonu ---
def get_website_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            print(f"URL '{url}' HTML içeriği döndürmedi (Tip: {content_type}). Atlanıyor.")
            return ""

        soup = BeautifulSoup(response.content, 'lxml')
        for script_or_style in soup(["script", "style", "header", "footer", "nav", "aside", "form", "noscript", "iframe", "button", "select", "input", "img", "svg", "link", "meta"]):
            script_or_style.decompose()
        
        body = soup.find('body')
        if body:
            paragraphs = body.find_all(['p', 'div', 'span', 'article', 'section', 'td', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            cleaned_text = ""
            for tag in paragraphs:
                text_content = tag.get_text(separator=' ', strip=True)
                if text_content and len(text_content.split()) > 2: # En az 3 kelime
                         cleaned_text += text_content + "\n\n"
            cleaned_text = "\n".join([line.strip() for line in cleaned_text.splitlines() if line.strip()])
            return cleaned_text
        return ""
    except requests.exceptions.RequestException as e:
        st.warning(f"Web sitesi içeriği çekilirken hata: {url} - {e}")
        return ""
    except Exception as e:
        st.warning(f"Web sitesi içeriği işlenirken beklenmedik hata: {url} - {e}")
        return ""

# # YENİ: Web Arama Fonksiyonu (Eğer 'ara:' özelliğini de istiyorsanız, önceki yanıttan alın)
# def search_web_for_query(query, num_results=3):
//     # ... (Önceki yanıttaki DDGS kodu) ...
//     pass

# # YENİ: Web İçeriği Özetleme için Prompt Şablonu (Eğer 'ara:' özelliğini de istiyorsanız)
# def get_web_summary_prompt_template():
//     # ... (Önceki yanıttaki prompt şablonu) ...
//     pass

# --- Diğer Yardımcı Fonksiyonlar ---
def get_pdf_text(pdf_docs):
    text = ""
    if pdf_docs:
        for pdf in pdf_docs:
            try:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text: text += page_text
            except Exception as e: st.warning(f"'{pdf.name}' dosyasından metin çıkarılırken hata: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""] # Noktayı da ayırıcı olarak ekledim
    )
    return text_splitter.split_text(text)

def create_vector_store_from_chunks(text_chunks, current_embeddings_model):
    if not text_chunks or not current_embeddings_model:
        return None
    try:
        # GÜNCELLEME: Metin parçaları boşsa veya sadece boşluk içeriyorsa FAISS hata verebilir.
        valid_chunks = [chunk for chunk in text_chunks if chunk.strip()]
        if not valid_chunks:
            st.warning("Vektör deposu oluşturmak için geçerli metin parçası bulunamadı.")
            return None
        return FAISS.from_texts(texts=valid_chunks, embedding=current_embeddings_model)
    except Exception as e:
        st.error(f"Vektör deposu oluşturulurken hata: {e}"); st.error(traceback.format_exc()); return None

def get_conversational_chain_prompt_template():
    prompt_template_str = """
    SENİN GÖREVİN: Sen, kullanıcı tarafından sağlanan bir metin kaynağındaki (bu bir PDF veya bir web sitesi içeriği olabilir) bilgilere dayanarak soruları yanıtlayan son derece dikkatli ve kuralcı bir AI asistanısın. Temel amacın, kullanıcının sorularına SADECE ve YALNIZCA aşağıda "Bağlam:" olarak belirtilen metin içeriğinden yararlanarak cevap vermektir.

    UYMAN GEREKEN KESİN KURALLAR:
    1.  BAĞLAM DIŞINA ÇIKMA: Kendi genel bilgini, internetten erişebileceğin diğer bilgileri veya daha önceki sohbetlerden edindiğin bilgileri KESİNLİKLE yanıtlarına dahil etme. Senin bilgi evrenin SADECE o anki "Bağlam:" ile sınırlıdır.
    2.  YORUMSUZ AKTARIM: "Bağlam:" içindeki bilgileri olduğu gibi veya anlamını değiştirmeden aktar. Kendi yorumlarını, çıkarımlarını, kişisel görüşlerini veya eklemelerini ASLA yapma.
    3.  BİLGİ YOKSA NET DURUŞ: Eğer kullanıcının sorusunun cevabı "Bağlam:" içinde AÇIKÇA bulunmuyorsa veya "Bağlam:" bölümü boş ise, TEREDDÜT ETMEDEN ve SADECE şu yanıtı ver:
        "Bu bilgi sağlanan kaynakta (PDF/Web Sitesi) bulunmuyor."
    4.  "BİLGİ YOK" MESAJININ KISITLARI: Yukarıdaki "bilgi yok" mesajını verdikten sonra, ASLA ek bir açıklama, özür, varsayım, yönlendirme veya "ama genel olarak..." gibi ifadeler kullanma. Cevabın SADECE bu standart mesajdan ibaret olmalıdır.
    5.  ÖZETLEME İSTEKLERİ: Eğer kullanıcı kaynağın genel bir özetini, ana konusunu, ne anlattığını veya benzeri bir genel bakış istiyorsa ("özetle", "ne hakkında", "konusu ne" gibi ifadelerle):
        a.  "Bağlam:" olarak sana sunulan (bu durumda genellikle kaynağın tamamı veya büyük bir kısmı olacaktır) metindeki ana fikirleri, kilit noktaları ve önemli detayları dikkatlice analiz et.
        b.  Bu analiz sonucunda, kaynağın genelini yansıtan, kapsamlı ama mümkün olduğunca öz bir özet oluştur.
        c.  Bu özeti oluştururken de KESİNLİKLE "Bağlam:" dışına çıkma. Sadece bağlamdaki bilgileri kullanarak özet yap. Eğer bağlam özet için yetersizse veya çok kısaysa, bunu belirt.
    6.  DOĞRUDAN VE NET OL: Cevapların açık, anlaşılır ve doğrudan sorulan soruyla ilgili olsun. Gereksiz uzun girişlerden veya dolaylı anlatımlardan kaçın.

    Bağlam:
    {context}

    Soru: {question}

    Cevap:"""
    return PromptTemplate(template=prompt_template_str, input_variables=["context", "question"])

# --- Session State ve Oturum Yönetimi ---
if "sessions" not in st.session_state: st.session_state.sessions = {}
if "current_session_id" not in st.session_state: st.session_state.current_session_id = None
if "prompt_template" not in st.session_state: st.session_state.prompt_template = get_conversational_chain_prompt_template()
# if "web_summary_prompt_template" not in st.session_state: st.session_state.web_summary_prompt_template = get_web_summary_prompt_template() # 'ara:' özelliği için

def create_new_session(session_type="pdf", session_name_prefix=None):
    session_id = str(uuid.uuid4())
    if session_name_prefix:
        session_name = f"{session_name_prefix} {len(st.session_state.sessions) + 1}"
    else:
        type_prefix = "PDF Sohbeti" if session_type == "pdf" else "Web Sohbeti"
        session_name = f"{type_prefix} {len(st.session_state.sessions) + 1}"

    st.session_state.sessions[session_id] = {
        "id": session_id, "name": session_name,
        "source_type": session_type,
        "source_info": None,
        "vector_store": None, "chat_history": [], "processed": False,
        "full_text_for_summary": None
    }
    st.session_state.current_session_id = session_id
    return session_id

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
            else: # Eğer hiç oturum kalmazsa varsayılan bir tane oluşturabiliriz
                 create_new_session() # veya None bırakıp kullanıcıyı yönlendirebiliriz. Şimdilik yeni oluşturuyor.
                 st.rerun()


st.title("🌐 Çok Kaynaklı AI Asistanı")

# --- Kenar Çubuğu ---
with st.sidebar:
    st.header("Sohbet Yönetimi")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Yeni PDF Sohbeti", key="new_pdf_chat", use_container_width=True): create_new_session(session_type="pdf"); st.rerun()
    with col2:
        if st.button("🔗 Yeni Web Sohbeti", key="new_web_chat", use_container_width=True): create_new_session(session_type="website"); st.rerun()

    session_options = {
        sid: f"{sdata['name']} ({sdata.get('source_info', 'Boş') if isinstance(sdata.get('source_info'), str) else ', '.join(sdata.get('source_info', [])) if sdata.get('source_info') else 'Boş'})"
        for sid, sdata in st.session_state.sessions.items()
    }

    if not session_options and st.session_state.current_session_id is None:
        create_new_session() # Uygulama ilk açıldığında varsayılan bir oturum oluştur
        st.rerun()

    if session_options:
        current_index = 0
        if st.session_state.current_session_id and st.session_state.current_session_id in session_options: # Kontrol eklendi
            current_index = list(session_options.keys()).index(st.session_state.current_session_id)
        elif session_options: # Eğer current_session_id geçerli değilse ama oturumlar varsa ilkini seç
            st.session_state.current_session_id = list(session_options.keys())[0]
            current_index = 0
            # st.rerun() # Bu döngüye sokabilir, dikkatli kullanılmalı. Şimdilik kapalı.
        
        selected_session_id = st.selectbox(
            "Aktif Sohbet:", options=list(session_options.keys()),
            format_func=lambda sid: session_options.get(sid, "Bilinmeyen Oturum"),
            index=current_index,
            key="session_selector"
        )
        if selected_session_id != st.session_state.current_session_id:
            st.session_state.current_session_id = selected_session_id; st.rerun()

        active_session = get_active_session_data()
        if active_session: # Burası artık daha güvenli olmalı
            st.markdown("---"); st.subheader(f"Düzenle: {active_session['name']}")
            raw_text_from_source = None # Bu değişkeni burada tanımlayalım

            # Sadece kenar çubuğundan PDF yükleme ve işleme
            if active_session["source_type"] == "pdf":
                uploader_key = f"pdf_uploader_{active_session['id']}"
                uploaded_files = st.file_uploader("PDF dosyaları:", accept_multiple_files=True, type="pdf", key=uploader_key, label_visibility="collapsed")
                if st.button("PDF'leri İşle", key=f"process_pdf_{active_session['id']}", use_container_width=True):
                    if uploaded_files:
                        active_session["source_info"] = [f.name for f in uploaded_files]
                        # raw_text_from_source aşağıda ortak blokta işlenecek
                        with st.spinner("PDF içeriği okunuyor..."):
                           raw_text_from_source = get_pdf_text(uploaded_files)
                        
                        # PDF işleme sonrası ortak blok
                        if raw_text_from_source is not None:
                            with st.spinner("Kaynak işleniyor... (PDF)"):
                                active_session["full_text_for_summary"] = raw_text_from_source
                                active_session["chat_history"] = [] # Kaynak değiştiğinde sohbeti temizle
                                active_session["vector_store"] = None
                                
                                if not raw_text_from_source.strip():
                                    st.error("PDF'ten metin çıkarılamadı veya boş.")
                                    active_session["processed"] = False; active_session["full_text_for_summary"] = None
                                else:
                                    text_chunks = get_text_chunks(raw_text_from_source)
                                    if not text_chunks:
                                        st.error("Metin parçalara ayrılamadı."); active_session["processed"] = False
                                    else:
                                        vector_store = create_vector_store_from_chunks(text_chunks, embeddings_model_global)
                                        if vector_store:
                                            active_session["vector_store"] = vector_store; active_session["processed"] = True
                                            st.success(f"PDF başarıyla işlendi."); st.rerun()
                                        else:
                                            st.error("Vektör deposu oluşturulamadı."); active_session["processed"] = False
                    else: st.warning("Lütfen PDF dosyası yükleyin.")
            
            # Kenar çubuğundan URL işleme (isteğe bağlı tutulabilir veya kaldırılabilir)
            elif active_session["source_type"] == "website":
                url_input_key = f"url_input_{active_session['id']}"
                # Değer olarak active_session'dan source_info'yu al, yoksa boş string
                current_url_val = active_session.get("source_info", "") if isinstance(active_session.get("source_info"), str) else ""
                website_url = st.text_input("Web sitesi URL'si (Kenar Çubuğu):", key=url_input_key, value=current_url_val, placeholder="https://ornek.com/sayfa", label_visibility="collapsed")
                
                # NOT: Kenar çubuğundan "Web Sitesini İşle" butonu artık birincil yöntem değil,
                # ana yöntem chat alanından `url:` komutu olacak. Bu buton yedek olarak kalabilir.
                if st.button("Web Sitesini İşle (Kenar Çubuğu)", key=f"process_web_sidebar_{active_session['id']}", use_container_width=True):
                    if website_url and website_url.startswith(("http://", "https://")):
                        active_session["source_info"] = website_url # URL'yi sakla
                        with st.spinner(f"İçerik çekiliyor: {website_url}"):
                            raw_text_from_source = get_website_text(website_url)
                        
                        # Web işleme sonrası ortak blok (sidebar için)
                        if raw_text_from_source is not None: # get_website_text artık hata durumunda None yerine "" dönebilir
                            with st.spinner("Kaynak işleniyor... (Web - Sidebar)"):
                                active_session["full_text_for_summary"] = raw_text_from_source
                                active_session["chat_history"] = []
                                active_session["vector_store"] = None
                                
                                if not raw_text_from_source.strip():
                                    st.error("Web sitesinden metin çıkarılamadı veya boş.")
                                    active_session["processed"] = False; active_session["full_text_for_summary"] = None
                                else:
                                    text_chunks = get_text_chunks(raw_text_from_source)
                                    if not text_chunks:
                                        st.error("Metin parçalara ayrılamadı."); active_session["processed"] = False
                                    else:
                                        vector_store = create_vector_store_from_chunks(text_chunks, embeddings_model_global)
                                        if vector_store:
                                            active_session["vector_store"] = vector_store; active_session["processed"] = True
                                            st.success(f"Web sitesi (sidebar) başarıyla işlendi."); st.rerun()
                                        else:
                                            st.error("Vektör deposu oluşturulamadı."); active_session["processed"] = False
                    elif website_url: st.warning("Lütfen geçerli bir URL girin.")
                    else: st.warning("Lütfen bir URL girin.")

            if active_session.get("processed") and active_session.get("source_info"):
                 source_display = active_session["source_info"]
                 if isinstance(source_display, list): source_display = ", ".join(source_display)
                 st.markdown(f"**İşlenen Kaynak:**")
                 st.caption(f"{source_display}")

            st.markdown("---")
            if st.button(f"'{active_session['name']}' Oturumunu Sil", type="secondary", key=f"delete_btn_{active_session['id']}", use_container_width=True):
                delete_session(active_session['id']); st.success(f"Oturum silindi."); st.rerun()
    else:
        st.sidebar.info("Henüz bir sohbet oturumu yok. Lütfen yeni bir tane oluşturun.")

# --- Ana Sohbet Alanı ---
active_session_data = get_active_session_data() # En güncel oturum bilgisini al
if active_session_data:
    st.header(f"Sohbet: {active_session_data['name']}")
    current_source_info = active_session_data.get("source_info")
    if current_source_info and active_session_data.get("processed"): # Sadece işlenmişse göster
        source_display = current_source_info
        if isinstance(source_display, list): source_display = ", ".join(source_display)
        st.caption(f"Mevcut Kaynak: {source_display}")
    # elif active_session_data.get("source_type") == "website" and not active_session_data.get("source_info"):
    #     st.caption(f"Bu web sohbeti için bir URL belirlemek üzere 'url: [adres]' yazın.")
    # elif not current_source_info:
    #     st.caption("Bu oturum için henüz bir kaynak işlenmedi.")

    # --- YENİ: Chat Komutları Tanımları ---
    # WEB_SEARCH_TRIGGER = "ara:" # Genel web arama için (bu örnekte kapalı)
    PROCESS_URL_TRIGGER = "url:" # Spesifik URL işleme için

    for message in active_session_data["chat_history"]:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    chat_input_placeholder = "Kaynak hakkında soru sorun..."
    if not active_session_data.get("processed"):
        if active_session_data["source_type"] == "pdf":
            chat_input_placeholder = "PDF'leri kenar çubuğundan yükleyip işleyin."
        elif active_session_data["source_type"] == "website":
            chat_input_placeholder = f"Bir web sitesini işlemek için '{PROCESS_URL_TRIGGER} [adres]' yazın."
        else: # Yeni, tipi belirsiz oturum
             chat_input_placeholder = f"PDF için kenar çubuğunu kullanın veya '{PROCESS_URL_TRIGGER} [adres]' ile web sitesi işleyin."
    else: # Kaynak işlenmişse
        chat_input_placeholder = f"İşlenmiş kaynak hakkında soru sorun veya '{PROCESS_URL_TRIGGER} [adres]' ile yeni URL işleyin."


    if user_query := st.chat_input(chat_input_placeholder, key=f"chat_input_{active_session_data['id']}"):
        active_session_data["chat_history"].append({"role": "user", "content": user_query})
        with st.chat_message("user"): st.markdown(user_query)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response_text = ""

            # --- YENİ: URL İşleme Komutu ---
            if user_query.lower().startswith(PROCESS_URL_TRIGGER):
                url_to_process = user_query[len(PROCESS_URL_TRIGGER):].strip()
                if url_to_process and url_to_process.startswith(("http://", "https://")):
                    message_placeholder.markdown(f"`{url_to_process}` adresinden içerik çekiliyor ve işleniyor...")
                    try:
                        raw_text_from_source = get_website_text(url_to_process)
                        if raw_text_from_source and raw_text_from_source.strip():
                            active_session_data["source_info"] = url_to_process
                            active_session_data["source_type"] = "website" # Oturum tipini de güncelle
                            active_session_data["full_text_for_summary"] = raw_text_from_source
                            
                            # Önceki sohbeti temizle, sadece bu komutu ve cevabını tutacak şekilde ayarla
                            # Bu komut zaten history'e eklendi, o yüzden sadece buraya kadar olanı alıp, asistan cevabını ekleyeceğiz.
                            # active_session_data["chat_history"] = [msg for msg in active_session_data["chat_history"] if msg["content"] == user_query and msg["role"] == "user"]
                            # YADA tamamen temizle ve yeni bir başlangıç yap:
                            active_session_data["chat_history"] = [
                                {"role": "user", "content": user_query}
                            ]

                            text_chunks = get_text_chunks(raw_text_from_source)
                            if not text_chunks:
                                full_response_text = "Web sitesinden metin parçalara ayrılamadı. Lütfen farklı bir URL deneyin veya sayfanın metin içeriği olduğundan emin olun."
                                active_session_data["processed"] = False
                            else:
                                vector_store = create_vector_store_from_chunks(text_chunks, embeddings_model_global)
                                if vector_store:
                                    active_session_data["vector_store"] = vector_store
                                    active_session_data["processed"] = True
                                    
                                    # Oturum adını URL'den gelen domain ile güncelle (isteğe bağlı iyileştirme)
                                    try:
                                        parsed_url = urlparse(url_to_process)
                                        domain = parsed_url.netloc
                                        if domain:
                                            active_session_data["name"] = f"Web: {domain.replace('www.','')}"
                                    except Exception as name_ex:
                                        print(f"URL'den isim alınamadı: {name_ex}")
                                        active_session_data["name"] = f"Web: {url_to_process[:30]}..."
                                    
                                    full_response_text = f"İçerik `{url_to_process}` adresinden başarıyla işlendi. Artık bu kaynak hakkında soru sorabilirsiniz."
                                    st.rerun() # Sayfayı yeniden yükle (sidebar ve chat'i güncellemek için)
                                else:
                                    full_response_text = "Web sitesi içeriği için vektör deposu oluşturulamadı. Metin parçaları geçerli olmayabilir."
                                    active_session_data["processed"] = False
                        else:
                            full_response_text = f"`{url_to_process}` adresinden metin çıkarılamadı veya boş içerik döndü. Lütfen URL'yi kontrol edin."
                            active_session_data["processed"] = False
                    except Exception as e:
                        st.error(f"URL işlenirken hata: {e}")
                        full_response_text = f"`{url_to_process}` işlenirken bir hata oluştu: {traceback.format_exc()}"
                        active_session_data["processed"] = False
                    message_placeholder.markdown(full_response_text)
                else:
                    full_response_text = "Lütfen geçerli bir URL girin (örneğin: url: https://ornek.com)."
                    message_placeholder.markdown(full_response_text)
            
            # --- Genel Web Arama Mantığı ('ara:' komutu - bu örnekte kapalı) ---
            # elif user_query.lower().startswith(WEB_SEARCH_TRIGGER):
            #     # ... (Eğer 'ara:' özelliği de isteniyorsa, önceki yanıttaki kod buraya gelecek) ...
            #     full_response_text = "Genel web arama özelliği ('ara:') bu örnekte aktif değil."
            #     message_placeholder.markdown(full_response_text)

            # --- Mevcut RAG (PDF/İşlenmiş Web Sitesi) Mantığı ---
            else:
                can_answer_from_source = active_session_data.get("processed", False) and \
                                         (active_session_data.get("vector_store") or active_session_data.get("full_text_for_summary"))
                
                if not can_answer_from_source:
                    # Bu mesaj zaten chat_input_placeholder'da veriliyor, ama bir fallback olarak kalabilir.
                    if active_session_data["source_type"] == "pdf":
                         full_response_text = "Lütfen önce kenar çubuğundan bu oturum için PDF dosyalarını yükleyip işleyin."
                    elif active_session_data["source_type"] == "website":
                         full_response_text = f"Lütfen önce kenar çubuğundan bu oturum için bir web sitesi URL'si girip işleyin veya sohbet alanına '{PROCESS_URL_TRIGGER} [URL]' yazarak yeni bir URL sağlayın."
                    else:
                        full_response_text = f"Lütfen önce bir kaynak sağlayın. PDF için kenar çubuğunu, web sitesi için '{PROCESS_URL_TRIGGER} [URL]' komutunu kullanın."
                    message_placeholder.markdown(full_response_text)
                else: # Kaynak işlenmiş ve soru soruluyor
                    try:
                        context_text = ""
                        summary_keywords = ["özet", "özetle", "ne anlatıyor", "konusu ne", "ana fikir", "genel olarak", "genel bakış", "kısaca", "summarize", "what is it about", "main idea", "overview", "gist", "tell me about this document"]
                        is_summary_request = any(keyword in user_query.lower() for keyword in summary_keywords)

                        if is_summary_request and active_session_data.get("full_text_for_summary"):
                            context_text = active_session_data["full_text_for_summary"]
                            MAX_CONTEXT_CHARS = 700000 
                            if len(context_text) > MAX_CONTEXT_CHARS:
                                context_text = context_text[:MAX_CONTEXT_CHARS] + "\n\n... (metin özet için çok uzundu ve kısaltıldı)"
                        
                        elif active_session_data.get("vector_store"): 
                            docs = active_session_data["vector_store"].similarity_search(query=user_query, k=5) 
                            if docs:
                                context_text = "\n\n".join([doc.page_content for doc in docs])
                        
                        if not context_text: 
                            if is_summary_request:
                                full_response_text = "Kaynak metni özetleme için çok kısa veya boş."
                            else:
                                full_response_text = "Bu bilgi sağlanan kaynakta (PDF/Web Sitesi) bulunmuyor."
                        else:
                            current_prompt_template = st.session_state.prompt_template
                            formatted_prompt = current_prompt_template.format(context=context_text, question=user_query)
                            
                            # DEBUG: st.text_area("LLM'e Gönderilen Prompt", formatted_prompt, height=300)

                            for chunk_resp in llm_client.stream(formatted_prompt):
                                if hasattr(chunk_resp, 'content'):
                                    full_response_text += chunk_resp.content
                                    message_placeholder.markdown(full_response_text + "▌")
                                else: 
                                    print(f"Beklenmedik chunk yapısı: {chunk_resp}")

                        if not full_response_text.strip() and not context_text: # Eğer LLM boş döndüyse ve context de yoksa
                             if not is_summary_request: # Soruya cevap bulunamadıysa
                                full_response_text = "Bu bilgi sağlanan kaynakta (PDF/Web Sitesi) bulunmuyor."

                        message_placeholder.markdown(full_response_text)

                    except Exception as e:
                        st.error(f"Yanıt alınırken bir hata oluştu: {e}"); st.error(traceback.format_exc())
                        full_response_text = "Üzgünüm, yanıt işlenirken bir hata oluştu."; message_placeholder.markdown(full_response_text)
            
            active_session_data["chat_history"].append({"role": "assistant", "content": full_response_text})
else:
    st.info("Lütfen kenar çubuğundan bir sohbet seçin veya yeni bir tane başlatın.")

st.sidebar.markdown("---")
st.sidebar.caption(f"LLM: {GOOGLE_LLM_MODEL_NAME}")
st.sidebar.caption(f"Embedding: {GOOGLE_EMBEDDING_MODEL_NAME}")

# --- END OF FILE app2.py ---
