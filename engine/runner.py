import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_MODEL_CACHE = {}

def _load_model(model_id):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. CPU fallback is disabled for this project.")
    if model_id not in _MODEL_CACHE:
        torch.cuda.empty_cache()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model.to("cuda")
        model.eval()
        _MODEL_CACHE[model_id] = (tokenizer, model)
    return _MODEL_CACHE[model_id]

def run_inference(model_id, prompt, max_new_tokens=64):
    tokenizer, model = _load_model(model_id)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    elapsed = time.perf_counter() - start
    generated = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    token_count = int(generated.shape[0])
    allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
    return {
        "text": text, "generated_tokens": token_count,
        "latency_seconds": round(elapsed, 4),
        "tokens_per_second": round(token_count / elapsed, 3) if elapsed else 0,
        "gpu_memory_allocated_gb": round(allocated, 3),
        "gpu_memory_reserved_gb": round(reserved, 3),
        "device": torch.cuda.get_device_name(0),
        "precision": "FP16"
    }
