# Setting Up Ollama for PreShip AI v2.0

PreShip AI v2.0 includes natural-language explanations for order risk assessments. These are powered by **Ollama**, a local LLM runtime that runs on your machine without requiring cloud APIs or API keys.

## Quick Start

### 1. Install Ollama

Download and install Ollama from: **https://ollama.ai**

Ollama runs on macOS, Linux, and Windows (preview).

### 2. Start the Ollama Server

Open a terminal and run:

```bash
ollama serve
```

You should see output like:

```
listening on 127.0.0.1:11434
```

Leave this terminal running. Ollama will listen on `http://localhost:11434`.

### 3. Pull the qwen2.5:7b Model

Open a second terminal and run:

```bash
ollama pull qwen2.5:7b
```

This downloads the qwen2.5:7b model (~5GB). It's a compact, fast 7-billion-parameter model that runs on modest hardware.

### 4. Run the PreShip AI Streamlit App

Once Ollama is running, start the Streamlit app:

```powershell
.\PreShipAIpython\Scripts\streamlit.exe run risk_manager_app.py
```

Or (if using a system Python environment):

```bash
streamlit run risk_manager_app.py
```

### 5. Check Ollama Status

In the sidebar under **LLM Settings**, the app will show:

- ✅ **Ollama is running** (if Ollama server is responding)
- ✅ **qwen2.5:7b model is loaded** (if the model is available)
- ⚠️ **Warnings** (if Ollama is not running or model is missing)

---

## Fallback Behavior

If Ollama is **not running**:

- LLM-generated summaries will use a deterministic template (still useful, but less natural-sounding).
- The app continues to work normally.
- Example fallback: "Flagged primarily due to Discount, Brand — mitigated by Product Rating."

If Ollama **is running** but the qwen2.5:7b model is not loaded:

- The app will suggest running: `ollama pull qwen2.5:7b`
- Fallback will be used until the model is loaded.

---

## System Requirements

- **CPU:** Modern multi-core processor (Intel, AMD, or Apple Silicon)
- **RAM:** 8GB+ recommended for qwen2.5:7b
- **Disk:** ~5GB for the model
- **Network:** Ollama runs locally; no internet required after model download

### GPU Acceleration (Optional)

Ollama can use NVIDIA CUDA or Metal (macOS) for faster inference:

- **NVIDIA (Windows/Linux):** Ollama auto-detects NVIDIA GPUs. Requires CUDA 11.8+.
- **Metal (macOS):** Ollama auto-detects Apple Silicon or Metal-capable GPUs.
- **CPU-only:** Works fine, slightly slower (~1-2 seconds per request).

---

## Troubleshooting

### Issue: Connection refused at http://localhost:11434

**Solution:** Make sure the Ollama server is running in a separate terminal:

```bash
ollama serve
```

### Issue: Model not found after `ollama pull`

**Solution:** Pull the model again:

```bash
ollama pull qwen2.5:7b
```

Check available models:

```bash
ollama list
```

### Issue: Slow response time

**Solution:** 
- Ensure sufficient RAM (8GB+).
- If using CPU-only, response times may be 5-10 seconds (normal).
- Enable GPU acceleration if available.

### Issue: Ollama conflicts with other services on port 11434

**Solution:** Edit `~/.ollama/ollama.config` (or `%APPDATA%\.ollama\ollama.config` on Windows) and change:

```
OLLAMA_HOST=127.0.0.1:11435
```

Then update the app's `ollama_url` parameter in the code if needed.

---

## Alternative Models

qwen2.5:7b is recommended for this use case (small, fast, good quality), but you can try:

- `mistral:7b` — Similar size, optimized for instruction-following.
- `neural-chat:7b` — Chat-optimized, good for conversational explanations.
- `llama2:7b` — Widely used, good baseline.

To switch models, simply pull and use another:

```bash
ollama pull mistral:7b
```

Then modify `generate_explanation_with_llm()` in `llm_explain.py` to use `"model": "mistral:7b"`.

---

## More Information

- **Ollama docs:** https://github.com/ollama/ollama
- **qwen2.5 model info:** https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- **Supported models:** https://ollama.ai/library

---

## Privacy & Cost

✅ **Privacy:** All LLM inference runs locally on your machine. No data is sent to cloud servers.

✅ **Cost:** Ollama and qwen2.5:7b are free and open-source.

✅ **Speed:** Local inference is typically 0.5–2 seconds per order (depending on CPU/GPU).

Enjoy transparent, cost-free, private risk explanations! 🎯
