# chatbotv2
# Çok Kaynaklı AI Asistanı / Multi-Source AI Assistant

Bu proje, Streamlit kullanılarak geliştirilmiş bir yapay zeka destekli asistandır. Kullanıcıların PDF dosyalarından veya belirli web sitesi URL'lerinden içerik yüklemesine ve bu içerikler hakkında sorular sormasına olanak tanır. Asistan, yanıtlarını yalnızca sağlanan kaynaklara dayandırır.

This project is an AI-powered assistant developed using Streamlit. It allows users to upload content from PDF files or specific website URLs and ask questions about that content. The assistant bases its answers solely on the provided sources.

---

##  Türkçe Açıklama

### Özellikler

*   **PDF Yükleme ve Sorgulama:** Birden fazla PDF dosyası yükleyebilir ve içerikleri hakkında sorular sorabilirsiniz.
*   **Web Sitesi URL'si ile Sorgulama:**
    *   Kenar çubuğundan "Yeni Web Sohbeti" oluşturup, ilgili alana bir URL girerek o web sitesinin içeriğini işleyebilirsiniz.
    *   Alternatif olarak, herhangi bir aktif sohbet sırasında chat alanına `url: https://ornek.com` şeklinde bir komut yazarak hızlıca bir web sitesini işleyebilirsiniz. Bu işlem, mevcut oturumun kaynağını belirtilen URL ile güncelleyecektir.
*   **Kaynak Odaklı Yanıtlar:** Yapay zeka, sorularınıza yalnızca yüklediğiniz PDF'lerin veya işlediğiniz web sitesinin içeriğini kullanarak yanıt verir.
*   **Oturum Yönetimi:** Farklı kaynaklar için ayrı sohbet oturumları oluşturabilir, bunlar arasında geçiş yapabilir ve oturumları silebilirsiniz.
*   **Özetleme:** Yüklenen veya işlenen kaynağın genel bir özetini isteyebilirsiniz (örneğin, "bu doküman ne hakkında?" veya "özetle").
*   **Akışkan Yanıtlar:** AI asistanının yanıtları, yazıyormuş gibi gerçek zamanlı olarak görüntülenir.

### Kurulum ve Çalıştırma

1.  **Depoyu Klonlayın (Opsiyonel):**
    ```bash
    git clone <depo_url'si>
    cd <proje_dizini>
    ```

2.  **Gerekli Kütüphaneleri Yükleyin:**
    Proje dizininde bir `requirements.txt` dosyası olduğunu varsayarsak:
    ```bash
    pip install -r requirements.txt
    ```
    Eğer `requirements.txt` yoksa, ana Python dosyasındaki (`app.py` veya `app2.py`) import bildirimlerine bakarak kütüphaneleri manuel olarak yükleyebilirsiniz:
    ```bash
    pip install streamlit PyPDF2 langchain langchain-google-genai langchain-community faiss-cpu beautifulsoup4 requests urllib3
    ```
    *(Not: `faiss-cpu` veya GPU'nuz varsa `faiss-gpu` kullanabilirsiniz.)*

3.  **API Anahtarını Ayarlayın:**
    *   Proje ana dizininde `.streamlit` adında bir klasör oluşturun.
    *   Bu klasörün içine `secrets.toml` adında bir dosya oluşturun.
    *   `secrets.toml` dosyasının içeriğini aşağıdaki gibi düzenleyin ve kendi Google API anahtarınızı girin:
        ```toml
        GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY_HERE"
        # Opsiyonel: Kullanmak istediğiniz model adlarını belirtebilirsiniz
        # GOOGLE_LLM_MODEL_NAME = "gemini-1.5-flash-latest"
        # GOOGLE_EMBEDDING_MODEL_NAME = "models/embedding-001"
        ```
    *   Google API anahtarınızı [Google AI Studio](https://aistudio.google.com/app/apikey) üzerinden alabilirsiniz.

4.  **Uygulamayı Çalıştırın:**
    Proje ana dizinindeyken terminalde aşağıdaki komutu çalıştırın (Python dosyanızın adı `app.py` veya `app2.py` ise ona göre düzenleyin):
    ```bash
    streamlit run app.py
    ```

### Kullanım

1.  Uygulama tarayıcınızda açıldığında, kenar çubuğunu kullanarak:
    *   **"Yeni PDF Sohbeti"** butonuna tıklayarak PDF tabanlı bir sohbet başlatın. Ardından PDF dosyalarınızı yükleyip "PDF'leri İşle" butonuna tıklayın.
    *   **"Yeni Web Sohbeti"** butonuna tıklayarak web sitesi tabanlı bir sohbet başlatın.
        *   Bu oturumda, kenar çubuğundaki URL giriş alanına bir web sitesi adresi yapıştırıp "Web Sitesini İşle (Kenar Çubuğu)" butonuna tıklayabilirsiniz.
2.  Herhangi bir sohbet oturumundayken, ana sohbet alanının altındaki giriş kutusunu kullanarak:
    *   Yüklenmiş/işlenmiş kaynak hakkında sorular sorun.
    *   Bir web sitesini hızlıca işlemek için `url: https://siteadresi.com` formatında komut girin. Bu, mevcut oturumun kaynağını bu URL ile güncelleyecektir.
    *   Kaynağın özetini almak için "özetle", "konusu ne" gibi ifadeler kullanın.
3.  Farklı sohbet oturumları arasında geçiş yapmak veya oturumları silmek için kenar çubuğundaki "Aktif Sohbet" seçicisini ve "Oturumu Sil" butonunu kullanın.

---

## English Description

### Features

*   **PDF Upload and Querying:** Upload multiple PDF files and ask questions about their content.
*   **Website URL Querying:**
    *   Create a "New Web Chat" from the sidebar and process a website's content by entering its URL in the respective field.
    *   Alternatively, during any active chat session, quickly process a website by typing a command like `url: https://example.com` into the chat input area. This will update the current session's source to the specified URL.
*   **Source-Focused Answers:** The AI responds to your questions using only the content from the PDFs you've uploaded or the website you've processed.
*   **Session Management:** Create separate chat sessions for different sources, switch between them, and delete sessions.
*   **Summarization:** Request a general summary of the uploaded or processed source (e.g., "what is this document about?" or "summarize").
*   **Streaming Responses:** AI assistant's responses are displayed in real-time, as if being typed.

### Setup and Running

1.  **Clone the Repository (Optional):**
    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```

2.  **Install Required Libraries:**
    Assuming there is a `requirements.txt` file in the project directory:
    ```bash
    pip install -r requirements.txt
    ```
    If `requirements.txt` is not available, you can install the libraries manually by looking at the import statements in the main Python file (`app.py` or `app2.py`):
    ```bash
    pip install streamlit PyPDF2 langchain langchain-google-genai langchain-community faiss-cpu beautifulsoup4 requests urllib3
    ```
    *(Note: You can use `faiss-cpu` or `faiss-gpu` if you have a GPU.)*

3.  **Set Up Your API Key:**
    *   Create a folder named `.streamlit` in the project's root directory.
    *   Inside this folder, create a file named `secrets.toml`.
    *   Edit the `secrets.toml` file with the following content, inserting your own Google API key:
        ```toml
        GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY_HERE"
        # Optional: You can specify the model names you want to use
        # GOOGLE_LLM_MODEL_NAME = "gemini-1.5-flash-latest"
        # GOOGLE_EMBEDDING_MODEL_NAME = "models/embedding-001"
        ```
    *   You can obtain your Google API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

4.  **Run the Application:**
    While in the project's root directory, run the following command in your terminal (adjust if your Python file is named `app.py` or `app2.py`):
    ```bash
    streamlit run app.py
    ```

### Usage

1.  Once the application opens in your browser, use the sidebar to:
    *   Start a PDF-based chat by clicking the **"New PDF Chat"** button. Then, upload your PDF files and click "Process PDFs".
    *   Start a website-based chat by clicking the **"New Web Chat"** button.
        *   In this session, you can paste a website address into the URL input field in the sidebar and click "Process Website (Sidebar)".
2.  Within any chat session, use the input box at the bottom of the main chat area to:
    *   Ask questions about the loaded/processed source.
    *   Quickly process a website by entering a command in the format `url: https://websiteaddress.com`. This will update the current session's source to this URL.
    *   Use phrases like "summarize" or "what is its topic" to get a summary of the source.
3.  Use the "Active Chat" selector and the "Delete Session" button in the sidebar to switch between different chat sessions or delete them.

---

### Katkıda Bulunma / Contributing

Katkı sağlamak isterseniz, lütfen bir "issue" açın veya bir "pull request" gönderin.
If you'd like to contribute, please open an issue or submit a pull request.
