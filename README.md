# QAWS — Quantization-Aware Weight Streaming

A local research prototype for measuring LLM inference latency and CUDA memory use.

## Current implementation status
- Implemented: FastAPI dashboard, configured model registry, FP16 CUDA inference, result JSON persistence, latency/token throughput and peak CUDA allocated/reserved memory.
- Not implemented in this presentation build: INT8/INT4/INT3/INT2 model execution, true chunk-wise CPU-to-GPU weight streaming, MPI distributed execution, token-by-token streaming.

The dashboard only offers FP16 so it does not imply that planned modes are implemented.

## Setup (Windows PowerShell)
Create/activate a virtual environment:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```
Install CUDA PyTorch separately (this matches a CUDA 12.1 setup):
```powershell
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```
Install the remaining dependencies:
```powershell
pip install -r requirements.txt
```
Run:
```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Open http://127.0.0.1:8000

## Notes
- CUDA is required; CPU fallback is deliberately disabled.
- Start with Qwen2.5-0.5B-Instruct on a 6 GB GPU.
- Load one model at a time. Larger models may exceed available VRAM.
- Some Llama/Gemma repositories require accepting their Hugging Face access terms and logging in.
- The first run downloads model weights; ensure internet access and disk space.
