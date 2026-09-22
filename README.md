# LinguaPop

> **A lightweight desktop assistant for reading & learning English.**

**LinguaPop** is a lightweight, all-in-one desktop companion designed to make reading, translating, and learning English effortless on Windows. With global hotkeys, you can instantly translate selected text, extract text from any screen region using OCR, and listen to natural AI neural pronunciations right beside your mouse cursor without breaking your workflow.

---

## Key Features

- **Instant Selection Translation:** Highlight text in any application (browser, PDF, Word, code editor) and press a hotkey to see translated results in a sleek popup.
- **Screen OCR Extraction:** Capture text from non-selectable areas (images, videos, scanned PDFs) with ultra-fast offline OCR.
- **Local Neural Text-to-Speech (TTS):** Hear native pronunciations from pretrained Piper ONNX voices loaded locally for fast offline inference.
- **Dedicated Pronunciation Hotkey:** Speak highlighted words or sentences directly on-the-fly.
- **Translation History:** Keep track of past translations with search, quick copy, and audio replay.
- **Long Text Chunking:** Automatically segments long passages into sentences to preserve translation quality.

---

## Technologies & Libraries Used

This project is optimized for speed and low resource consumption using open-source technologies:

- **Neural Machine Translation (NMT):** [MarianMT](https://huggingface.co/docs/transformers/model_doc/marian) (`opus-mt-en-vi`) fine-tuned on the MTet dataset and academic/IELTS parallel corpora.
- **Inference Engine:** [CTranslate2](https://github.com/OpenNMT/CTranslate2) with 8-bit quantization (`int8`) for near-instant CPU/GPU inference.
- **Optical Character Recognition (OCR):** [RapidOCR](https://github.com/RapidAI/RapidOCR) (`rapidocr_onnxruntime`), combined with `OpenCV` and `Pillow`.
- **Text-to-Speech (TTS):** [Piper](https://github.com/rhasspy/piper) pretrained ONNX voices, with [pygame](https://github.com/pygame/pygame) for in-memory audio playback.
- **User Interface (UI):** Pure `tkinter` with custom dark styling and DPI awareness for crisp rendering on high-resolution displays.
- **System Interaction:** `pynput` for global hotkey hooks, `pyperclip` for clipboard integration.

---

## Installation & Setup with Virtual Environment (venv)

Ensure you have **Python 3.8 to 3.11** installed.

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
  *(If you get a script execution policy restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first)*

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

## Step 1: Convert the Model to CTranslate2

After downloading or fine-tuning your MarianMT model (e.g., in a `final_model/` directory):

```bash
python scripts/convert_to_ct2.py --model path/to/final_model
```

This command converts the model into optimized CTranslate2 format and creates:
- `models/en-vi-ct2/`
- `models/tokenizer-en-vi/`

---

## Step 2: Run the App

Launch LinguaPop with:

```bash
python src/main.py
```

---

## Shortcuts & Controls

| Shortcut / Key | Action | Description |
| :--- | :--- | :--- |
| **`Ctrl + Shift + Y`** | **Translate Highlighted Text** | Grabs selected text, translates EN ➔ VI, and displays the popup. |
| **`Ctrl + Shift + H`** | **OCR Screen & Translate** | Drag a bounding box on your screen to extract and translate text. |
| **`Ctrl + Alt + S`** | **Pronounce Selection** | Instantly pronounces selected text out loud. |

### In-Popup Controls:
- **`E`** or **`S`**: Pronounce source text (English).
- **`V`** or **`Space`**: Pronounce translated text (Vietnamese).
- **`Esc`**: Close popup and immediately stop audio playback.
- **Click on translated text**: Copies text to clipboard and closes popup.
- **Click "History"**: Opens the Translation History manager.

---

## Configuration (`config.yaml`)

You can customize hotkeys, TTS voices, reading speeds, and UI properties without modifying source code:

```yaml
hotkeys:
  translate_selection: "ctrl+shift+y"
  translate_ocr: "ctrl+shift+h"
  speak_selection: "ctrl+alt+s"

device: "cpu"          # "cpu" or "cuda"

engines:
  default: "en-vi"
  en-vi:
    model_dir: "models/en-vi-ct2"
    tokenizer_dir: "models/tokenizer-en-vi"

translation:
  max_input_length: 128
  num_beams: 4
  num_hypotheses: 1
  max_decode_length: 128

ui:
  popup_timeout_ms: 15000  # Auto-close timeout (0 = keep open)
  font_size: 13
  max_width: 480

ocr:
  lang: "en"
  use_gpu: false

tts:
  enabled: true
  engine: "piper"          # Local pretrained Piper voices
  auto_speak: false        # Auto-speak upon translation
  rate: "+0%"              # Speech speed: "+0%", "+10%", "-10%"
  volume: "+0%"
  voices:
    en: "models/tts/en_US-lessac-medium.onnx"
    vi: "models/tts/vi_VN-vais1000-medium.onnx"
```

---

## License

Distributed under the **MIT License**. Free for academic, personal, and commercial use.

---

## Acknowledgements

- **Models:** [Helsinki-NLP/opus-mt-en-vi](https://huggingface.co/Helsinki-NLP/opus-mt-en-vi), MTET Parallel Corpus.
- **Inference & NMT:** [CTranslate2](https://github.com/OpenNMT/CTranslate2), [Marian NMT](https://github.com/marian-nmt/marian), [Hugging Face Transformers](https://github.com/huggingface/transformers).
- **OCR:** [RapidOCR](https://github.com/RapidAI/RapidOCR).
- **Audio & TTS:** [Piper](https://github.com/rhasspy/piper), [pygame](https://github.com/pygame/pygame).
