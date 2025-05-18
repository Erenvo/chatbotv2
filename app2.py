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
from duckduckgo_search import DDGS # YENİ: DuckDuckGo arama için
# import trafilatura

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
        temperature=0.2, # Özetleme için biraz daha yüksek olabilir
        # safety_settings={ ... }
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
        # YENİ: İçerik tipini kontrol et (isteğe bağlı ama iyi bir pratik)
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            # st.warning(f"URL '{url}' HTML içeriği döndürmedi (Tip: {content_type}). Atlanıyor.")
            print(f"URL '{url}' HTML içeriği döndürmedi (Tip: {content_type}). Atlanıyor.")
            return "" # Veya None, işleme mantığınıza göre

        soup = BeautifulSoup(response.content, 'lxml')
        for script_or_style in soup(["script", "style", "header", "footer", "nav", "aside", "form", "noscript", "iframe", "button", "select", "input", "img", "svg", "link", "meta"]): # Daha fazla gereksiz etiket
            script_or_style.decompose()
        
        body = soup.find('body')
        if body:
            # text_nodes = body.find_all(string=True)
            # visible_text = ""
            # for t_node in text_nodes:
            #     parent = t_node.parent
            #     if parent.name not in ['style', 'script', 'head', 'title', 'meta', '[document]'] and not isinstance(t_node, Comment):
            #         stripped_text = t_node.strip()
            #         if stripped_text:
            #             visible_text += stripped_text + "\n"
            # cleaned_text = "\n".join([line.strip() for line in visible_text.splitlines() if line.strip()])
            
            # Daha iyi bir metin çıkarma yöntemi:
            paragraphs = body.find_all(['p', 'div', 'span', 'article', 'section', 'td', 'li']) # Daha fazla potansiyel metin içeren etiket
            cleaned_text = ""
            for tag in paragraphs:
                text_content = tag.get_text(separator=' ', strip=True)
                if text_content: # Sadece anlamlı içeriği olanları ekle
                    # Kısa, anlamsız stringleri filtrele (örneğin, sadece navigasyon linkleri vb.)
                    # Bu kısım daha da geliştirilebilir.
                    if len(text_content.split()) > 3: # En az 3 kelime varsa al gibi bir basit kural
                         cleaned_text += text_content + "\n\n" # Paragraflar arasına boşluk

            # Çoklu boşlukları ve satırları tek bir taneye indirge (son bir temizlik)
            cleaned_text = "\n".join([line.strip() for line in cleaned_text.splitlines() if line.strip()])
            return cleaned_text
        return ""
    except requests.exceptions.RequestException as e:
        st.warning(f"Web sitesi içeriği çekilirken hata: {url} - {e}") # st.error yerine warning
        return "" # Boş string döndür, akışı kesmesin
    except Exception as e:
        st.warning(f"Web sitesi içeriği işlenirken beklenmedik hata: {url} - {e}") # st.error yerine warning
        return "" # Boş string döndür

# YENİ: Web Arama Fonksiyonu
def search_web_for_query(query, num_results=3):
    """
    Verilen sorgu için web'de arama yapar ve ilk N sonucun URL'lerini döndürür.
    """
    results = []
    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, region='wt-wt', safesearch='moderate', max_results=num_results*2)) # Biraz fazla alıp filtreleyelim
            # Sadece geçerli URL'leri ve kısa olanları filtrele (örneğin PDF, DOC linkleri değil)
            count = 0
            for r in search_results:
                if r.get('href') and r['href'].startswith('http') and \
                   not any(ext in r['href'].lower() for ext in ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.zip', '.rar']):
                    results.append(r['href'])
                    count += 1
                    if count >= num_results:
                        break
            if not results and search_results: # Hiç uygun URL bulunamadıysa ilkini almayı dene
                if search_results[0].get('href'):
                     results.append(search_results[0]['href'])
    except Exception as e:
        st.error(f"Web araması sırasında hata: {e}")
    return results

# YENİ: Web İçeriği Özetleme için Prompt Şablonu
def get_web_summary_prompt_template():
    prompt_template_str = """
    SENİN GÖREVİN: Kullanıcının bir sorgusu üzerine web'den aşağıdaki "Bağlam:" bölümünde çeşitli sayfalardan bilgiler toplandı.
    Bu bilgileri kullanarak, kullanıcının orijinal sorgusuna yanıt veren, anlaşılır, tarafsız ve kapsamlı bir özet oluştur.
    Özetin SADECE sağlanan "Bağlam:" içindeki bilgilere dayanmalıdır.
    Eğer sağlanan bağlam yetersizse veya konuyla alakasızsa, "Sağlanan web sayfalarından bu sorgu için yeterli bilgi çıkarılamadı." yanıtını ver.
    Yanıtını doğrudan özetle başlat.

    Bağlam:
    {context}

    Kullanıcının Orijinal Sorgusu: {original_query}

    Özet Cevap:"""
    return PromptTemplate(template=prompt_template_str, input_variables=["context", "original_query"])

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
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def create_vector_store_from_chunks(text_chunks, current_embeddings_model):
    if not text_chunks or not current_embeddings_model:
        return None
    try:
        return FAISS.from_texts(texts=text_chunks, embedding=current_embeddings_model)
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
        c.  Bu özeti oluştururken de KESİNLİKLE "Bağlam:" dışına çıkma. Sadece bağlamdaki bilgileri kullanarak özet yap.
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
if "web_summary_prompt_template" not in st.session_state: st.session_state.web_summary_prompt_template = get_web_summary_prompt_template() # YENİ

def create_new_session(session_type="pdf"):
    session_id = str(uuid.uuid4())
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
        create_new_session(); st.rerun() # Varsayılan olarak bir PDF sohbeti oluşturabilir veya kullanıcıya seçtirebilir.

    if session_options:
        current_index = 0
        if st.session_state.current_session_id in session_options:
            current_index = list(session_options.keys()).index(st.session_state.current_session_id)
        
        selected_session_id = st.selectbox(
            "Aktif Sohbet:", options=list(session_options.keys()),
            format_func=lambda sid: session_options.get(sid, "Bilinmeyen"),
            index=current_index,
            key="session_selector"
        )
        if selected_session_id != st.session_state.current_session_id:
            st.session_state.current_session_id = selected_session_id; st.rerun()

        active_session = get_active_session_data()
        if active_session:
            st.markdown("---"); st.subheader(f"Düzenle: {active_session['name']}")
            raw_text_from_source = None

            if active_session["source_type"] == "pdf":
                uploader_key = f"pdf_uploader_{active_session['id']}"
                uploaded_files = st.file_uploader("PDF dosyaları:", accept_multiple_files=True, type="pdf", key=uploader_key, label_visibility="collapsed")
                if st.button("PDF'leri İşle", key=f"process_pdf_{active_session['id']}", use_container_width=True):
                    if uploaded_files:
                        active_session["source_info"] = [f.name for f in uploaded_files]
                        raw_text_from_source = get_pdf_text(uploaded_files)
                    else: st.warning("Lütfen PDF dosyası yükleyin.")
            
            elif active_session["source_type"] == "website":
                url_input_key = f"url_input_{active_session['id']}"
                website_url = st.text_input("Web sitesi URL'si:", key=url_input_key, value=active_session.get("source_info",""), placeholder="https://ornek.com/sayfa", label_visibility="collapsed")
                if st.button("Web Sitesini İşle", key=f"process_web_{active_session['id']}", use_container_width=True):
                    if website_url and website_url.startswith(("http://", "https://")):
                        active_session["source_info"] = website_url
                        with st.spinner(f"İçerik çekiliyor..."):
                            raw_text_from_source = get_website_text(website_url)
                    elif website_url: st.warning("Lütfen geçerli bir URL girin.")
                    else: st.warning("Lütfen bir URL girin.")

            if raw_text_from_source is not None:
                with st.spinner("Kaynak işleniyor..."):
                    active_session["full_text_for_summary"] = raw_text_from_source
                    active_session["chat_history"] = []
                    active_session["vector_store"] = None
                    
                    if not raw_text_from_source.strip():
                        st.error("Kaynak boş veya metin çıkarılamadı.")
                        active_session["processed"] = False; active_session["full_text_for_summary"] = None
                    else:
                        text_chunks = get_text_chunks(raw_text_from_source)
                        if not text_chunks:
                            st.error("Metin parçalara ayrılamadı."); active_session["processed"] = False
                        else:
                            vector_store = create_vector_store_from_chunks(text_chunks, embeddings_model_global)
                            if vector_store:
                                active_session["vector_store"] = vector_store; active_session["processed"] = True
                                st.success(f"Kaynak başarıyla işlendi."); st.rerun()
                            else:
                                st.error("Vektör deposu oluşturulamadı."); active_session["processed"] = False
            
            if active_session.get("processed") and active_session.get("source_info"):
                 source_display = active_session["source_info"]
                 if isinstance(source_display, list): source_display = ", ".join(source_display)
                 st.markdown(f"**İşlenen Kaynak:**")
                 st.caption(f"{source_display}")


            st.markdown("---")
            if st.button(f"'{active_session['name']}' Oturumunu Sil", type="secondary", key=f"delete_btn_{active_session['id']}", use_container_width=True):
                delete_session(active_session['id']); st.success(f"Oturum silindi."); st.rerun()
    else:
        st.sidebar.info("Henüz bir sohbet oturumu yok.")

# --- Ana Sohbet Alanı ---
active_session_data = get_active_session_data()
if active_session_data:
    st.header(f"Sohbet: {active_session_data['name']}")
    current_source_info = active_session_data.get("source_info")
    if current_source_info:
        source_display = current_source_info
        if isinstance(source_display, list): source_display = ", ".join(source_display)
        st.caption(f"Mevcut Kaynak: {source_display}")
    # YENİ: Web arama komutu için bilgi
    # else:
    #     st.caption("Bu oturum için henüz bir kaynak işlenmedi. Web'de arama yapmak için 'ara: [sorgunuz]' yazabilirsiniz.")
    
    # YENİ: Web arama tetikleyici
    WEB_SEARCH_TRIGGER = "ara:" 
    # veya st.checkbox("Web'de Ara") gibi bir UI elemanı da kullanabilirsiniz.

    for message in active_session_data["chat_history"]:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if user_query := st.chat_input(f"Kaynak hakkında soru sorun veya '{WEB_SEARCH_TRIGGER} [sorgunuz]' ile web'de arayın..."):
        active_session_data["chat_history"].append({"role": "user", "content": user_query})
        with st.chat_message("user"): st.markdown(user_query)
        with st.chat_message("assistant"):
            message_placeholder = st.empty(); full_response_text = ""
            
            # YENİ: Web Arama Mantığı
            if user_query.lower().startswith(WEB_SEARCH_TRIGGER):
                search_term = user_query[len(WEB_SEARCH_TRIGGER):].strip()
                if not search_term:
                    full_response_text = "Lütfen aramak istediğiniz konuyu belirtin (örneğin: ara: Türkiye'nin başkenti)."
                    message_placeholder.markdown(full_response_text)
                else:
                    message_placeholder.markdown(f"'{search_term}' için web'de arama yapılıyor ve özetleniyor...")
                    try:
                        search_urls = search_web_for_query(search_term, num_results=3) # İlk 3 sonucu alalım
                        if not search_urls:
                            full_response_text = f"'{search_term}' için web'de sonuç bulunamadı."
                        else:
                            st.caption(f"Bulunan kaynaklar: {', '.join(search_urls)}") # DEBUG: Hangi URL'ler bulundu
                            all_web_text = ""
                            MAX_WEB_CONTENT_PER_PAGE = 10000 # Her sayfadan alınacak max karakter (çok uzun sayfalardan kaçınmak için)
                            MAX_TOTAL_WEB_CONTENT = 50000 # LLM'e gönderilecek toplam max karakter

                            for i, url in enumerate(search_urls):
                                with st.spinner(f"'{url}' adresinden içerik çekiliyor... ({i+1}/{len(search_urls)})"):
                                    page_text = get_website_text(url)
                                if page_text:
                                    all_web_text += f"\n\n--- {url} sayfasından içerik ---\n\n" + page_text[:MAX_WEB_CONTENT_PER_PAGE]
                                if len(all_web_text) > MAX_TOTAL_WEB_CONTENT:
                                    all_web_text = all_web_text[:MAX_TOTAL_WEB_CONTENT] + "\n\n... (içerik çok uzundu ve kısaltıldı)"
                                    break
                            
                            if not all_web_text.strip():
                                full_response_text = "Web sayfalarından anlamlı içerik çıkarılamadı."
                            else:
                                # DEBUG: st.text_area("Web'den Çekilen Toplam Metin", all_web_text, height=200)
                                web_summary_prompt = st.session_state.web_summary_prompt_template.format(
                                    context=all_web_text, 
                                    original_query=search_term
                                )
                                # DEBUG: st.text_area("Web Özetleme Prompt'u", web_summary_prompt, height=200)
                                for chunk in llm_client.stream(web_summary_prompt):
                                    if hasattr(chunk, 'content'):
                                        full_response_text += chunk.content
                                        message_placeholder.markdown(full_response_text + "▌")
                                if not full_response_text: # LLM boş döndürürse
                                    full_response_text = "Sağlanan web sayfalarından bu sorgu için yeterli bilgi çıkarılamadı."
                    except Exception as e:
                        st.error(f"Web araması ve özetleme sırasında hata: {e}"); st.error(traceback.format_exc())
                        full_response_text = "Üzgünüm, web araması sırasında bir hata oluştu."
                    message_placeholder.markdown(full_response_text)
            
            # Mevcut RAG (PDF/Web Sitesi) mantığı
            else:
                can_answer_from_source = active_session_data.get("processed", False) and \
                                         (active_session_data.get("vector_store") or active_session_data.get("full_text_for_summary"))
                
                if not can_answer_from_source:
                    full_response_text = "Lütfen önce kenar çubuğundan bu oturum için bir kaynak (PDF/Web Sitesi) yükleyip işleyin veya web'de arama yapmak için 'ara: [sorgunuz]' komutunu kullanın."
                    message_placeholder.markdown(full_response_text)
                else:
                    try:
                        context_text = ""
                        summary_keywords = ["özetle", "ne anlatıyor", "konusu ne", "ana fikir", "genel olarak", "genel bakış", "kısaca", "summarize", "what is it about", "main idea", "overview", "gist", "tell me about this document"]
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
                            full_response_text = "Bu bilgi sağlanan kaynakta (PDF/Web Sitesi) bulunmuyor."
                        else:
                            current_prompt_template = st.session_state.prompt_template
                            formatted_prompt = current_prompt_template.format(context=context_text, question=user_query)
                            
                            for chunk in llm_client.stream(formatted_prompt):
                                if hasattr(chunk, 'content'):
                                    full_response_text += chunk.content
                                    message_placeholder.markdown(full_response_text + "▌")
                                else: 
                                    print(f"Beklenmedik chunk yapısı: {chunk}")

                        message_placeholder.markdown(full_response_text)

                    except Exception as e:
                        st.error(f"Yanıt alınırken bir hata oluştu: {e}"); st.error(traceback.format_exc())
                        full_response_text = "Üzgünüm, bir hata oluştu."; message_placeholder.markdown(full_response_text)
            
            active_session_data["chat_history"].append({"role": "assistant", "content": full_response_text})
else:
    st.info("Lütfen kenar çubuğundan bir sohbet seçin veya yeni bir tane başlatın.")

st.sidebar.markdown("---")
st.sidebar.caption(f"LLM: {GOOGLE_LLM_MODEL_NAME}")
st.sidebar.caption(f"Embedding: {GOOGLE_EMBEDDING_MODEL_NAME}")
