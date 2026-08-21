# Desktop Translator (EN->VI)

A convenient Desktop translation tool (English to Vietnamese). Translate highlighted text anywhere on your Windows system using a global hotkey, and the result will instantly appear in a small popup near your mouse cursor. Additionally, the tool supports Optical Character Recognition (OCR) to extract and translate text from screenshots.

## Technologies & Libraries Used

This project is built and optimized for speed using open-source technologies:

- **Translation Model:** [MarianMT](https://huggingface.co/docs/transformers/model_doc/marian) (`opus-mt-en-vi`) fine-tuned on the MTet dataset and Academic (custom by myself).
- **Inference Engine:** [CTranslate2](https://github.com/OpenNMT/CTranslate2) combined with Hugging Face `transformers` for ultra-fast and lightweight inference (int8 quantization).
- **Optical Character Recognition (OCR):** [RapidOCR](https://github.com/RapidAI/RapidOCR) (`rapidocr_onnxruntime` version), combined with `OpenCV` (`cv2`) and `Pillow` for image processing and text extraction.
- **Text-to-Speech (TTS):** [edge-tts](https://github.com/rany2/edge-tts) for high-quality Microsoft Edge Neural AI speech synthesis, [pygame](https://github.com/pygame/pygame) for smooth asynchronous audio playback, and [pyttsx3](https://github.com/nateshmbhat/pyttsx3) for 100% offline Windows SAPI5 fallback.
- **User Interface (UI):** `tkinter` for smooth popup windows and screen selection overlays.
- **System Interaction:** `pynput` for capturing global hotkeys, `pyperclip` for clipboard operations, and `ctypes` for DPI awareness (preventing blurriness on high-resolution screens).

## Installation & Setup with Virtual Environment (venv)

Ensure you have Python 3.8 to 3.11 installed.

### 1. Create a Virtual Environment (`venv`)

Open your terminal in the project root directory and run:

```bash
# Windows
python -m venv venv
```

### 2. Activate the Virtual Environment

- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If you get a script execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first)*

- **Windows (Command Prompt / cmd.exe):**
  ```cmd
  venv\Scripts\activate.bat
  ```

- **Windows (Git Bash) / Linux / macOS:**
  ```bash
  source venv/Scripts/activate  # on Windows Git Bash
  # source venv/bin/activate    # on Linux/macOS
  ```

### 3. Install Dependencies

Once the virtual environment is activated `(venv)`, install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Step 1: Convert the Model

After downloading the output from Kaggle or your trained model and extracting it (e.g., a `final_model/` directory):

```bash
python scripts/convert_to_ct2.py --model path/to/final_model
```

This command will convert the model and create the necessary configuration directories: `models/en-vi-ct2/` and `models/tokenizer-en-vi/`.

## Step 2: Run the App

Launch the application with the following command (make sure `venv` is activated):

```bash
python src/main.py
```

### Usage
- **Text Translation:** Highlight any text → press **Ctrl+Shift+Y** → the translation popup will appear.
- **OCR Translation:** Press **Ctrl+Shift+H** → Drag mouse to select a screen region to OCR & translate.
- **Text-to-Speech (Pronounce):** Highlight any text → press **Ctrl+Alt+S** → pronounce & read aloud instantly.
- **Popup Controls:**
  - Press `E` / `S`: Pronounce source text (English).
  - Press `V` / `Space`: Pronounce translated text (Vietnamese).
  - Press `Esc`: Close popup and stop audio.
- **Exit:** Close the running terminal window to exit the app.

## Extended Features

- **Hybrid Text-to-Speech (TTS):** Microsoft Edge Neural AI TTS with automatic offline `pyttsx3` fallback.
- **Long Text Translation:** The application automatically splits long paragraphs into individual sentences for smooth translation, preventing truncation at the 128-token limit.
- **Multi-Engine Support:** The `EngineManager` system allows for easy configuration of additional translation directions (VI→EN) or different models.

## License

This project is distributed under the **MIT License**. You are free to use, modify, and distribute it.

## Acknowledgements

This project is built upon several open-source projects and resources:

### Models

* MarianMT (English ↔ Vietnamese Neural Machine Translation)
* Base model: Helsinki-NLP/opus-mt-en-vi
* Fine-tuned on MTET and additional academic-domain parallel corpora

### Open-source Libraries

* Hugging Face Transformers
* CTranslate2
* PyTorch
* RapidOCR
* OpenCV
* Pillow
* edge-tts
* pygame
* pyttsx3
* pynput
* pyperclip

### References

* Marian NMT: https://github.com/marian-nmt/marian
* Hugging Face Transformers: https://github.com/huggingface/transformers
* CTranslate2: https://github.com/OpenNMT/CTranslate2
* RapidOCR: https://github.com/RapidAI/RapidOCR
* edge-tts: https://github.com/rany2/edge-tts
* pygame: https://github.com/pygame/pygame
* pyttsx3: https://github.com/nateshmbhat/pyttsx3

Special thanks to the open-source community for providing the tools and resources that made this project possible.
