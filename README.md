# Arkani Quran ASR - RunPod Deployment

NeMo FastConformer model fine-tuned on Quranic recitation, deployed as a serverless API.

## Files
- `handler.py` — RunPod serverless handler (receives audio, returns text)
- `Dockerfile` — Container definition (based on NVIDIA NeMo)
- `requirements.txt` — Python dependencies

## Deployment Steps

### Step 1: Upload Model to Hugging Face
The model (`arkani_quran_model.nemo`, ~459MB) is too large for GitHub.
Upload it to Hugging Face instead:

```bash
# Install HF CLI
pip install huggingface_hub

# Login (get token from https://huggingface.co/settings/tokens)
huggingface-cli login

# Create repo and upload
huggingface-cli repo create arkani-quran-asr --type model
huggingface-cli upload arkani-quran-asr ../arkani_quran_model.nemo
```

### Step 2: Update handler.py
Change `YOUR_HF_USERNAME` to your actual Hugging Face username in:
- `handler.py` line 12
- `Dockerfile` line 16

### Step 3: Push deploy folder to GitHub
```bash
cd deploy
git init
git add .
git commit -m "Arkani Quran ASR - RunPod deployment"
git remote add origin https://github.com/YOUR_USERNAME/arkani-quran-asr-api.git
git push -u origin main
```

### Step 4: Connect to RunPod
1. Go to RunPod → Serverless → Connect GitHub
2. Select your repo
3. Choose GPU: **A40** or **L4** (recommended for NeMo)
4. Set Environment Variables:
   - `HF_REPO_ID` = `your-hf-username/arkani-quran-asr`
   - `HF_FILENAME` = `arkani_quran_model.nemo`
5. Click **Create Endpoint**

### Step 5: Test the API
```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"audio": "<base64-wav-data>"}}'
```
