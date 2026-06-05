"""
llm_client.py — LLM Integration cho Legal RAG

Hỗ trợ 2 chế độ:
1. Dev mode: Model nhỏ (7B Q4), context 2-4K, iterate nhanh
2. Demo mode: Model tốt hơn, streaming, UX mượt

Backends:
- llama.cpp (local)
- OpenRouter API
- OpenAI-compatible API
- HuggingFace Transformers
"""

import os
import json
import time
import random
import subprocess
import threading
from typing import Optional, List, Dict, Generator, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import urllib.request
import urllib.error


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL RATE LIMITER — giới hạn tần suất gọi API, tránh 429
# ═══════════════════════════════════════════════════════════════════════════════

class _GlobalRateLimiter:
    """
    Token-bucket rate limiter dùng chung cho MỌI API backend.
    
    Đảm bảo mỗi request API cách nhau ít nhất `min_interval` giây.
    Thread-safe.
    """
    
    def __init__(self, min_interval: float = 1.5):
        self.min_interval = min_interval  # seconds giữa 2 lần gọi API
        self._last_call_time = 0.0
        self._lock = threading.Lock()
        self._total_waits = 0
        self._total_wait_time = 0.0
    
    def wait(self):
        """Block cho đến khi được phép gọi API tiếp."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                self._total_waits += 1
                self._total_wait_time += sleep_time
                time.sleep(sleep_time)
            self._last_call_time = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_waits": self._total_waits,
            "total_wait_time_s": round(self._total_wait_time, 1),
            "min_interval": self.min_interval,
        }


# Singleton — dùng chung cho tất cả LLMClient instances
_rate_limiter = _GlobalRateLimiter(min_interval=1.5)


class LLMBackend(Enum):
    """Các backend LLM được hỗ trợ."""
    LLAMA_CPP = "llama_cpp"             # Local llama.cpp
    OPENROUTER = "openrouter"           # OpenRouter API
    OPENAI = "openai"                   # OpenAI API
    GEMINI = "gemini"                   # Google Gemini
    QWEN = "qwen"                       # Qwen3 via OpenRouter (fallback khi Gemini hết quota)
    HUGGINGFACE = "huggingface"         # HuggingFace Transformers
    PLACEHOLDER = "placeholder"         # Placeholder for testing


class LLMMode(Enum):
    """Chế độ hoạt động."""
    DEV = "dev"         # Development: nhỏ, nhanh
    DEMO = "demo"       # Demo: tốt hơn, streaming


@dataclass
class LLMConfig:
    """Cấu hình LLM."""
    
    # Backend selection
    backend: LLMBackend = LLMBackend.PLACEHOLDER
    mode: LLMMode = LLMMode.DEV
    
    # Model paths/names
    model_path: Optional[str] = None    # Path cho local models
    model_name: str = "default"         # Model name cho API
    
    # Generation parameters
    max_tokens: int = 512
    temperature: float = 0.1            # Low for legal accuracy
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    
    # Context
    context_length: int = 4096          # Dev: 2-4K, Demo: 8K+
    
    # API settings
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    require_json: bool = True           # Bắt buộc trả về JSON (hỗ trợ Gemini/OpenAI JSON mode)
    
    # Timeout
    timeout_seconds: int = 60
    
    # llama.cpp specific
    n_gpu_layers: int = -1              # -1 = all layers on GPU
    n_threads: int = 4
    
    # Streaming
    enable_streaming: bool = False


@dataclass
class LLMResponse:
    """Response từ LLM."""
    
    text: str
    finish_reason: str = "stop"
    tokens_used: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None
    raw_response: Optional[Dict] = None
    
    # ── Anti-429 metadata ──
    api_failed: bool = False             # True nếu API gọi thất bại (429/5xx/timeout)
    fallback_to_local: bool = False      # True nếu đã fallback sang local
    retry_count: int = 0                 # Số lần retry trước khi thành công/thất bại


def select_local_model_for_vram(reserved_vram_gb: float = 1.0) -> str:
    """
    Tự động chọn Qwen model phù hợp với VRAM còn lại.
    
    Tính toán: available = total - reserved (cho bi-encoder + cross-encoder ~1GB)
    
    Kiểm tra transformers version:
    - >= 4.49: dùng Qwen3 (mới, tốt hơn)
    - >= 4.37: dùng Qwen2.5 (tương thích rộng)
    
    4-bit quantization VRAM estimates:
        8B  → ~5.0GB
        4B  → ~2.5GB
        1.5-1.7B → ~1.2GB
        0.5-0.6B → ~0.5GB
    
    Returns:
        Model ID (HuggingFace Hub)
    """
    # Xác định Qwen series dựa trên transformers version
    try:
        import transformers
        ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        if ver >= (4, 49):
            # Qwen3 architecture supported
            models_by_vram = {
                6.0: "Qwen/Qwen3-8B",
                3.0: "Qwen/Qwen3-4B",
                1.5: "Qwen/Qwen3-1.7B",
                0.0: "Qwen/Qwen3-0.6B",
            }
            series = "Qwen3"
        else:
            # Fallback to Qwen2.5 (transformers >= 4.37)
            models_by_vram = {
                6.0: "Qwen/Qwen2.5-7B-Instruct",
                3.0: "Qwen/Qwen2.5-3B-Instruct",
                1.5: "Qwen/Qwen2.5-1.5B-Instruct",
                0.0: "Qwen/Qwen2.5-0.5B-Instruct",
            }
            series = "Qwen2.5"
            print(f"[AutoModel] transformers {transformers.__version__} < 4.49 → using {series} (upgrade for Qwen3)")
    except Exception:
        models_by_vram = {
            3.0: "Qwen/Qwen2.5-3B-Instruct",
            1.5: "Qwen/Qwen2.5-1.5B-Instruct",
            0.0: "Qwen/Qwen2.5-0.5B-Instruct",
        }
        series = "Qwen2.5"
    
    try:
        import torch
        if not torch.cuda.is_available():
            # CPU: dùng model nhỏ nhất
            model = list(models_by_vram.values())[-1]
            print(f"[AutoModel] No GPU → {model}")
            return model
        
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        available = total_vram - reserved_vram_gb
        
        # Chọn model lớn nhất vừa với VRAM
        model = list(models_by_vram.values())[-1]  # smallest as default
        for min_vram, model_id in sorted(models_by_vram.items(), reverse=True):
            if available >= min_vram:
                model = model_id
                break
        
        print(f"[AutoModel] VRAM: {total_vram:.1f}GB total, ~{available:.1f}GB available → {model}")
        return model
    except Exception:
        return f"Qwen/{series}-1.5B-Instruct" if "2.5" in series else f"Qwen/{series}-1.7B"


class LLMClient:
    """
    Unified LLM Client cho Legal RAG.
    
    Auto-detect backend từ environment hoặc config.
    
    Anti-429 features:
    - Global rate limiter (min 1.5s giữa 2 lần gọi API)
    - Retry + exponential backoff + jitter (5 lần, 2→4→8→16→32s)
    - Fallback sang local_client khi API fail sau tất cả retries
    """
    
    # Retry config
    MAX_RETRIES = 5
    BASE_DELAY = 2.0       # seconds
    JITTER_FACTOR = 0.2    # ±20%
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        fallback_client: Optional['LLMClient'] = None
    ):
        self.config = config or self._auto_detect_config()
        self.fallback_client = fallback_client  # Dùng khi API fails
        self._llama_process = None
        self._initialize_backend()
        
        # Stats tracking
        self._api_fail_count = 0
        self._fallback_count = 0
    
    def _auto_detect_config(self) -> LLMConfig:
        """Auto-detect config từ environment."""
        
        config = LLMConfig()
        
        # Check API keys
        if os.environ.get("GEMINI_API_KEY"):
            config.backend = LLMBackend.GEMINI
            config.api_key = os.environ["GEMINI_API_KEY"]
            config.model_name = "gemini-2.5-flash"
        elif os.environ.get("OPENROUTER_API_KEY"):
            # Qwen3 free-tier qua OpenRouter
            config.backend = LLMBackend.QWEN
            config.api_key = os.environ["OPENROUTER_API_KEY"]
            config.model_name = "qwen/qwen3-30b-a3b:free"
        elif os.environ.get("OPENAI_API_KEY"):
            config.backend = LLMBackend.OPENAI
            config.api_key = os.environ["OPENAI_API_KEY"]
            config.model_name = "gpt-3.5-turbo"
        else:
            # Check for local llama.cpp model
            model_paths = [
                Path("llama.cpp/models"),
                Path("models"),
                Path.home() / ".cache/llama.cpp/models"
            ]
            for mp in model_paths:
                if mp.exists():
                    gguf_files = list(mp.glob("*.gguf"))
                    if gguf_files:
                        config.backend = LLMBackend.LLAMA_CPP
                        config.model_path = str(gguf_files[0])
                        break
            
            if config.backend == LLMBackend.PLACEHOLDER:
                print("[LLMClient] No API key or local model found. Using placeholder mode.")
        
        return config
    
    def _initialize_backend(self):
        """Initialize các backends cần setup."""
        
        if self.config.backend == LLMBackend.LLAMA_CPP:
            self._check_llama_cpp()
        elif self.config.backend == LLMBackend.HUGGINGFACE:
            self._init_huggingface()
    
    def _check_llama_cpp(self):
        """Check llama.cpp installation."""
        # Check if llama-cli or main exists
        llama_paths = [
            "llama.cpp/build/bin/Release/main.exe",
            "llama.cpp/build/bin/main",
            "llama.cpp/main",
            "llama-cli",
        ]
        
        self._llama_executable = None
        for path in llama_paths:
            if Path(path).exists() or self._which(path):
                self._llama_executable = path
                break
        
        if not self._llama_executable:
            print("[LLMClient] Warning: llama.cpp executable not found")
    
    def _which(self, program: str) -> Optional[str]:
        """Find executable in PATH."""
        import shutil
        return shutil.which(program)
    
    @staticmethod
    def _guess_model_size_gb(model_id: str) -> float:
        """Estimate model size in billions of params from model name."""
        import re as _re
        # Match patterns like "1.5B", "3B", "7B", "0.5B", "1.7B"
        m = _re.search(r'(\d+(?:\.\d+)?)\s*[Bb]', model_id)
        if m:
            return float(m.group(1))
        # Fallback: assume medium model
        return 3.0
    
    def _init_huggingface(self):
        """Initialize HuggingFace transformers (4-bit quantization + chat template)."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            model_id = self.config.model_path or self.config.model_name
            
            # Default model nếu chưa chỉ định
            if model_id == "default":
                model_id = "Qwen/Qwen2.5-3B-Instruct"
                print(f"[HF] No model specified, using default: {model_id}")
            
            print(f"[HF] Loading model: {model_id}...")
            
            # Load tokenizer
            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
            if self._hf_tokenizer.pad_token is None:
                self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token
            
            # ── Quantization & dtype selection ──
            # Strategy: 
            #   Small models (≤ 3B): fp16 trực tiếp (đủ VRAM, tránh bnb compat issues)
            #   Large models (> 3B): 4-bit quantization nếu bitsandbytes available
            load_kwargs = {
                "device_map": "auto",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            use_4bit = False
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"[HF] GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram_gb:.1f}GB")
                
                # Estimate model size from name
                model_size_hint = self._guess_model_size_gb(model_id)
                fp16_size = model_size_hint * 2  # fp16 ≈ 2 bytes/param
                
                # Small model (<= 2B) and plenty of VRAM -> fp16 direct
                # Threshold reduced to avoid OOM on 4GB laptops
                if model_size_hint <= 2.0 and fp16_size <= (vram_gb - 1.5):
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    load_kwargs["torch_dtype"] = dtype
                    print(f"[HF] Small model ({model_size_hint:.1f}B) → fp16 direct (~{fp16_size:.1f}GB VRAM)")
                else:
                    # Large model — try 4-bit quantization
                    try:
                        from transformers import BitsAndBytesConfig
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                        )
                        load_kwargs["quantization_config"] = bnb_config
                        use_4bit = True
                        print(f"[HF] Large model ({model_size_hint:.1f}B) → 4-bit NF4 quantization")
                    except ImportError:
                        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                        load_kwargs["torch_dtype"] = dtype
                        print(f"[HF] bitsandbytes not installed → using fp16 (may OOM for large models)")
                        print(f"[HF] TIP: pip install bitsandbytes")
            else:
                load_kwargs["torch_dtype"] = torch.float32
                print("[HF] No GPU, using fp32 (rất chậm)")
            
            # Load model (with fallback: 4-bit → fp16 nếu bnb gặp lỗi compat)
            try:
                self._hf_model = AutoModelForCausalLM.from_pretrained(
                    model_id, **load_kwargs
                )
            except Exception as load_err:
                if use_4bit:
                    print(f"[HF] 4-bit loading failed ({load_err}), retrying with fp16...")
                    # Retry without quantization
                    load_kwargs.pop("quantization_config", None)
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    load_kwargs["torch_dtype"] = dtype
                    use_4bit = False
                    self._hf_model = AutoModelForCausalLM.from_pretrained(
                        model_id, **load_kwargs
                    )
                else:
                    raise
            
            try:
                self._hf_model.eval()
            except Exception:
                pass  # .eval() không tương thích với một số quantized models
            
            # Check chat template
            self._hf_has_chat_template = (
                hasattr(self._hf_tokenizer, 'chat_template') and
                self._hf_tokenizer.chat_template is not None
            )
            
            # Report
            param_count = sum(p.numel() for p in self._hf_model.parameters()) / 1e9
            quant_str = "4-bit NF4" if use_4bit else str(load_kwargs.get("torch_dtype", "auto"))
            print(f"[HF] ✓ Model loaded: {param_count:.1f}B params, {quant_str}, chat_template={self._hf_has_chat_template}")
            
        except ImportError as e:
            print(f"[LLMClient] Missing dependency: {e}")
            print(f"[LLMClient] Install: pip install transformers torch accelerate bitsandbytes")
            self.config.backend = LLMBackend.PLACEHOLDER
        except Exception as e:
            print(f"[LLMClient] Failed to load HF model: {e}")
            self.config.backend = LLMBackend.PLACEHOLDER
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GENERATION METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    @property
    def _is_api_backend(self) -> bool:
        """Check if current backend calls external API (cần rate limit + retry)."""
        return self.config.backend in (
            LLMBackend.OPENROUTER, LLMBackend.OPENAI, LLMBackend.GEMINI, LLMBackend.QWEN
        )
    
    def _compute_delay(self, attempt: int) -> float:
        """
        Exponential backoff + jitter.
        
        attempt 0 → 2.0s ± 20%
        attempt 1 → 4.0s ± 20%
        attempt 2 → 8.0s ± 20%
        attempt 3 → 16.0s ± 20%
        attempt 4 → 32.0s ± 20%
        """
        base = self.BASE_DELAY * (2 ** attempt)
        jitter = base * self.JITTER_FACTOR * (2 * random.random() - 1)  # ±20%
        return max(0.5, base + jitter)
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Kiểm tra error có nên retry không (429, 5xx, timeout)."""
        if isinstance(error, urllib.error.HTTPError):
            return error.code in (429, 500, 502, 503, 504)
        if isinstance(error, (urllib.error.URLError, TimeoutError, ConnectionError)):
            return True
        return False
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response từ LLM — với retry + rate limit + fallback.
        
        Flow:
        1. Rate limit (nếu API backend)
        2. Gọi backend
        3. Nếu 429/5xx → retry với exponential backoff + jitter (tối đa 5 lần)
        4. Nếu vẫn fail → fallback sang local_client (nếu có)
        5. Nếu fallback cũng fail → trả error
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            **kwargs: Override generation params
        
        Returns:
            LLMResponse (với api_failed / fallback_to_local metadata)
        """
        
        start_time = time.time()
        is_api = self._is_api_backend
        
        # ── Rate limit cho API backends ──
        if is_api:
            _rate_limiter.wait()
        
        # ── Dispatch to backend (local backends: no retry) ──
        if not is_api:
            try:
                response = self._dispatch_backend(prompt, system_prompt, **kwargs)
                response.latency_ms = (time.time() - start_time) * 1000
                return response
            except Exception as e:
                return LLMResponse(
                    text="",
                    error=str(e),
                    latency_ms=(time.time() - start_time) * 1000
                )
        
        # ── API backends: retry loop ──
        last_error = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                response = self._dispatch_backend(prompt, system_prompt, **kwargs)
                response.retry_count = attempt
                response.latency_ms = (time.time() - start_time) * 1000
                
                # Check if response is actually an error (some backends return error in response)
                if response.error and attempt < self.MAX_RETRIES:
                    # Check if it looks like a retryable error
                    if any(code in (response.error or "") for code in ["429", "500", "502", "503", "504"]):
                        delay = self._compute_delay(attempt)
                        print(f"[LLM] {self.config.backend.value} error: {response.error[:80]} "
                              f"— retry {attempt+1}/{self.MAX_RETRIES} in {delay:.1f}s")
                        time.sleep(delay)
                        _rate_limiter.wait()  # Re-acquire rate limit token
                        continue
                    # HTTP 400 / blocked = non-retryable → skip retries, go to fallback
                    if any(s in (response.error or "").lower() for s in ["400", "blocked", "invalid", "403"]):
                        print(f"[LLM] {self.config.backend.value} non-retryable: {response.error[:120]}")
                        break  # Nhảy thẳng tới fallback
                
                return response
                
            except Exception as e:
                last_error = e
                if self._is_retryable_error(e) and attempt < self.MAX_RETRIES:
                    delay = self._compute_delay(attempt)
                    error_code = getattr(e, 'code', type(e).__name__)
                    print(f"[LLM] {self.config.backend.value} {error_code} "
                          f"— retry {attempt+1}/{self.MAX_RETRIES} in {delay:.1f}s")
                    time.sleep(delay)
                    _rate_limiter.wait()  # Re-acquire rate limit token
                    continue
                break
        
        # ── All retries exhausted → try fallback ──
        self._api_fail_count += 1
        error_msg = str(last_error) if last_error else "API failed after max retries"
        print(f"[LLM] ✗ {self.config.backend.value} failed after {self.MAX_RETRIES} retries: {error_msg[:100]}")
        
        if self.fallback_client is not None:
            self._fallback_count += 1
            print(f"[LLM] ↪ Fallback → {self.fallback_client.config.backend.value}")
            fallback_response = self.fallback_client.generate(prompt, system_prompt, **kwargs)
            fallback_response.api_failed = True
            fallback_response.fallback_to_local = True
            fallback_response.retry_count = self.MAX_RETRIES
            fallback_response.latency_ms = (time.time() - start_time) * 1000
            return fallback_response
        
        # No fallback available
        return LLMResponse(
            text="",
            error=error_msg,
            api_failed=True,
            fallback_to_local=False,
            retry_count=self.MAX_RETRIES,
            latency_ms=(time.time() - start_time) * 1000
        )
    
    def _truncate_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> tuple:
        """
        Truncate prompt + system_prompt để không vượt quá context_length.
        Ước lượng 1.5 chars/token cho tiếng Việt.
        Giữ system_prompt nguyên, cắt context trong user prompt nếu cần.
        """
        max_input_tokens = self.config.context_length - self.config.max_tokens - 100  # buffer
        max_input_chars = int(max_input_tokens * 1.5)
        
        sys_len = len(system_prompt) if system_prompt else 0
        total_len = sys_len + len(prompt)
        
        if total_len <= max_input_chars:
            return prompt, system_prompt
        
        # Cắt user prompt, giữ system prompt
        allowed_prompt_chars = max_input_chars - sys_len
        if allowed_prompt_chars < 200:
            # System prompt quá dài, cắt cả hai
            system_prompt = (system_prompt or "")[:max_input_chars // 2]
            allowed_prompt_chars = max_input_chars // 2
        
        if len(prompt) > allowed_prompt_chars:
            # Cắt thông minh: giữ câu hỏi (cuối prompt), cắt context (đầu prompt)
            # Tìm phần "## CÂU HỎI" trong prompt
            question_marker = prompt.rfind("## CÂU HỎI")
            if question_marker > 0:
                question_part = prompt[question_marker:]  # Giữ nguyên
                context_part = prompt[:question_marker]
                remaining = allowed_prompt_chars - len(question_part)
                if remaining > 100:
                    prompt = context_part[:remaining] + "\n[...context đã rút gọn...]\n\n" + question_part
                else:
                    prompt = prompt[:allowed_prompt_chars]
            else:
                prompt = prompt[:allowed_prompt_chars]
        
        return prompt, system_prompt
    
    def _dispatch_backend(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Route to the correct backend implementation (no retry logic here)."""
        # Truncate prompt trước khi gửi API — tránh HTTP 400 do prompt quá dài
        if self._is_api_backend:
            prompt, system_prompt = self._truncate_prompt(prompt, system_prompt)
        
        if self.config.backend == LLMBackend.PLACEHOLDER:
            return self._generate_placeholder(prompt)
        elif self.config.backend == LLMBackend.LLAMA_CPP:
            return self._generate_llama_cpp(prompt, system_prompt, **kwargs)
        elif self.config.backend == LLMBackend.OPENROUTER:
            return self._generate_openrouter(prompt, system_prompt, **kwargs)
        elif self.config.backend == LLMBackend.OPENAI:
            return self._generate_openai(prompt, system_prompt, **kwargs)
        elif self.config.backend == LLMBackend.GEMINI:
            return self._generate_gemini(prompt, system_prompt, **kwargs)
        elif self.config.backend == LLMBackend.QWEN:
            return self._generate_qwen(prompt, system_prompt, **kwargs)
        elif self.config.backend == LLMBackend.HUGGINGFACE:
            return self._generate_huggingface(prompt, system_prompt, **kwargs)
        else:
            return self._generate_placeholder(prompt)
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        callback: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Streaming generation cho demo mode.
        
        Yields:
            Text chunks as they are generated
        """
        
        if self.config.backend == LLMBackend.OPENROUTER:
            yield from self._stream_openrouter(prompt, system_prompt, **kwargs)
        elif self.config.backend == LLMBackend.OPENAI:
            yield from self._stream_openai(prompt, system_prompt, **kwargs)
        else:
            # Fallback: generate full then yield
            response = self.generate(prompt, system_prompt, **kwargs)
            yield response.text
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BACKEND IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _generate_placeholder(self, prompt: str) -> LLMResponse:
        """Placeholder response cho testing."""
        return LLMResponse(
            text=json.dumps({
                "answer": "[Placeholder] Đây là câu trả lời placeholder cho mục đích testing.",
                "citations": [],
                "abstain": False,
                "reason": None
            }, ensure_ascii=False),
            finish_reason="placeholder",
            tokens_used=0
        )
    
    def _generate_llama_cpp(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate using llama.cpp."""
        
        if not self._llama_executable or not self.config.model_path:
            return self._generate_placeholder(prompt)
        
        # Build full prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Build command
        cmd = [
            self._llama_executable,
            "-m", self.config.model_path,
            "-p", full_prompt,
            "-n", str(kwargs.get("max_tokens", self.config.max_tokens)),
            "--temp", str(kwargs.get("temperature", self.config.temperature)),
            "--top-p", str(kwargs.get("top_p", self.config.top_p)),
            "--ctx-size", str(self.config.context_length),
            "-ngl", str(self.config.n_gpu_layers),
            "-t", str(self.config.n_threads),
            "--no-display-prompt"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds
            )
            
            return LLMResponse(
                text=result.stdout.strip(),
                finish_reason="stop" if result.returncode == 0 else "error",
                error=result.stderr if result.returncode != 0 else None
            )
            
        except subprocess.TimeoutExpired:
            return LLMResponse(
                text="",
                finish_reason="timeout",
                error="Generation timed out"
            )
    
    def _generate_openrouter(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate using OpenRouter API."""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        body = json.dumps({
            "model": kwargs.get("model", self.config.model_name),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }).encode()
        
        req = urllib.request.Request(url, data=body, headers={
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/legal-chatbot",
            "X-Title": "Legal Chatbot"
        })
        
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as r:
                resp = json.loads(r.read())
                
                content = resp["choices"][0]["message"]["content"]
                usage = resp.get("usage", {})
                
                return LLMResponse(
                    text=content,
                    finish_reason=resp["choices"][0].get("finish_reason", "stop"),
                    tokens_used=usage.get("total_tokens", 0),
                    raw_response=resp
                )
                
        except urllib.error.HTTPError as e:
            return LLMResponse(
                text="",
                error=f"HTTP Error: {e.code} - {e.read().decode()}"
            )
        except urllib.error.URLError as e:
            return LLMResponse(
                text="",
                error=f"URL Error: {e.reason}"
            )
    
    def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate using OpenAI API."""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        url = self.config.api_base_url or "https://api.openai.com/v1/chat/completions"
        body = json.dumps({
            "model": kwargs.get("model", self.config.model_name),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }).encode()
        
        req = urllib.request.Request(url, data=body, headers={
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        })
        
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as r:
                resp = json.loads(r.read())
                
                content = resp["choices"][0]["message"]["content"]
                usage = resp.get("usage", {})
                
                return LLMResponse(
                    text=content,
                    finish_reason=resp["choices"][0].get("finish_reason", "stop"),
                    tokens_used=usage.get("total_tokens", 0),
                    raw_response=resp
                )
                
        except Exception as e:
            return LLMResponse(text="", error=str(e))
    
    def _generate_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate using Google Gemini API (retry handled by generate())."""
        
        # ── Safety settings: tắt block cho legal content ──
        safety_settings = [
            {"category": cat, "threshold": "BLOCK_NONE"}
            for cat in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH", 
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            ]
        ]
        
        # Build request body — use systemInstruction if available
        contents = [{"parts": [{"text": prompt}]}]
        body_dict = {
            "contents": contents,
            "safetySettings": safety_settings,
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "responseMimeType": "application/json" if self.config.require_json else "text/plain",
            }
        }
        if system_prompt:
            body_dict["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        
        # Thử model names theo thứ tự ưu tiên
        model_name = self.config.model_name or "gemini-2.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.config.api_key}"
        body = json.dumps(body_dict, ensure_ascii=False).encode("utf-8")
        
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json; charset=utf-8"
        })
        
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as r:
                resp = json.loads(r.read())
                
                # Check for blocked/empty responses
                candidates = resp.get("candidates", [])
                if not candidates:
                    prompt_feedback = resp.get("promptFeedback", {})
                    block_reason = prompt_feedback.get("blockReason", "unknown")
                    return LLMResponse(text="", error=f"Gemini blocked: {block_reason}", raw_response=resp)
                
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                
                # Gemini can also return a 'refusal' field at the candidate level in some versions
                refusal = candidates[0].get("refusal")
                if refusal:
                    return LLMResponse(text=json.dumps({"answer": refusal, "abstain": True, "reason": "refusal"}), finish_reason="stop", raw_response=resp)

                if not parts:
                    finish_reason = candidates[0].get("finishReason", "unknown")
                    # Check for safety ratings if blocked
                    safety_ratings = candidates[0].get("safetyRatings", [])
                    safety_blocked = any(r.get("probability") != "NEGLIGIBLE" for r in safety_ratings)
                    error_msg = f"Gemini empty response: finishReason={finish_reason}"
                    if safety_blocked:
                        error_msg += " (possibly safety blocked)"
                    return LLMResponse(text="", error=error_msg, raw_response=resp)
                
                # DEBUG: xem structure parts
                # print(f"[Gemini DEBUG] num_parts={len(parts)}")
                
                # Gemini 2.5 Flash / 2.0 Pro trả multi-part response:
                #   parts[n] = {"thought": true, "text": "..."} (thinking)
                #   parts[m] = {"text": "..."} (answer thực)
                text_parts = []
                for part in parts:
                    # Bỏ qua thinking/thought parts - check cả 'thought' key và content
                    if part.get("thought") is True:
                        continue
                    
                    text = part.get("text", "")
                    if text:
                        # Tránh lấy nhầm phần trích dẫn nếu model lặp lại context (hiếm gặp với system prompt tốt)
                        text_parts.append(text)
                
                if not text_parts:
                    # Nếu vẫn không có text nào (ví dụ model chỉ trả về thought), lấy part cuối cùng coi như fallback
                    content_str = parts[-1].get("text", "") if parts else ""
                else:
                    content_str = "\n".join(text_parts)
                
                return LLMResponse(
                    text=content_str,
                    finish_reason="stop",
                    raw_response=resp
                )
                
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode()[:500]
            except Exception:
                pass
            error_msg = f"Gemini HTTP {e.code}: {error_body}"
            print(f"[Gemini] {error_msg[:150]}")
            
            # HTTP 400 thường do content bị block hoặc model name sai
            # Mark as retryable for 429/5xx, non-retryable for 400/403
            return LLMResponse(text="", error=error_msg)
        except Exception as e:
            return LLMResponse(text="", error=str(e))
    
    def _generate_qwen(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate using Qwen3 via OpenRouter API.
        
        Fallback khi Gemini hết quota / lỗi.
        Free-tier models: qwen/qwen3-30b-a3b:free, qwen/qwen3-8b:free
        Paid: qwen/qwen3-235b-a22b
        """
        
        api_key = self.config.api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            return LLMResponse(text="", error="OPENROUTER_API_KEY not set for Qwen backend")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Model selection: prefer free-tier cho dev
        model = self.config.model_name or "qwen/qwen3-30b-a3b:free"
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        body = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }, ensure_ascii=False).encode("utf-8")
        
        req = urllib.request.Request(url, data=body, headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
            "HTTP-Referer": "https://github.com/legal-chatbot",
            "X-Title": "Legal Chatbot"
        })
        
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as r:
                resp = json.loads(r.read())
                
                content = resp["choices"][0]["message"]["content"]
                usage = resp.get("usage", {})
                
                # Qwen3 enable thinking — strip <think>...</think> tags nếu có
                import re as _re
                content = _re.sub(r'<think>[\s\S]*?</think>\s*', '', content).strip()
                
                return LLMResponse(
                    text=content,
                    finish_reason=resp["choices"][0].get("finish_reason", "stop"),
                    tokens_used=usage.get("total_tokens", 0),
                    raw_response=resp
                )
                
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode()[:300]
            except Exception:
                pass
            return LLMResponse(text="", error=f"Qwen HTTP {e.code}: {error_body}")
        except Exception as e:
            return LLMResponse(text="", error=str(e))
    
    def _generate_huggingface(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate using HuggingFace Transformers.
        
        Hỗ trợ:
        - Chat template (Vistral, Mistral, Llama-2 format)
        - Fallback: raw prompt nếu không có chat template
        """
        import torch
        
        if not hasattr(self, '_hf_model') or not hasattr(self, '_hf_tokenizer'):
            return self._generate_placeholder(prompt)
        
        try:
            max_new_tokens = kwargs.get("max_tokens", self.config.max_tokens)
            temperature = kwargs.get("temperature", self.config.temperature)
            
            # Build input — prefer chat template
            # Generation prefix: seed Vietnamese JSON to prevent English output
            gen_prefix = kwargs.pop("generation_prefix", '{"answer": "Theo ')
            
            if self._hf_has_chat_template:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                input_text = self._hf_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                # Append generation prefix to force Vietnamese JSON start
                if gen_prefix:
                    input_text += gen_prefix
            else:
                # Fallback: concat system + user
                if system_prompt:
                    input_text = f"{system_prompt}\n\n{prompt}\n\n### Trả lời:\n"
                else:
                    input_text = f"{prompt}\n\n### Trả lời:\n"
                if gen_prefix:
                    input_text += gen_prefix
            
            # Tokenize
            inputs = self._hf_tokenizer(
                input_text, return_tensors="pt", truncation=True,
                max_length=self.config.context_length - max_new_tokens
            ).to(self._hf_model.device)
            input_len = inputs["input_ids"].shape[1]
            
            # Generate
            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": temperature > 0,
                    "pad_token_id": self._hf_tokenizer.pad_token_id,
                }
                if temperature > 0:
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_p"] = kwargs.get("top_p", self.config.top_p)
                    gen_kwargs["top_k"] = kwargs.get("top_k", self.config.top_k)
                    gen_kwargs["repetition_penalty"] = kwargs.get(
                        "repeat_penalty", self.config.repeat_penalty
                    )
                
                outputs = self._hf_model.generate(**inputs, **gen_kwargs)
            
            # Decode only new tokens
            new_tokens = outputs[0][input_len:]
            text = self._hf_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            # Prepend generation prefix so JSON is complete
            if gen_prefix:
                text = gen_prefix + text
            
            return LLMResponse(
                text=text,
                finish_reason="stop",
                tokens_used=len(new_tokens),
            )
            
        except torch.cuda.OutOfMemoryError:
            import gc; gc.collect(); torch.cuda.empty_cache()
            return LLMResponse(text="", error="CUDA OOM — reduce context_length or max_tokens")
        except Exception as e:
            return LLMResponse(text="", error=str(e))
    
    def _stream_openrouter(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Streaming from OpenRouter."""
        
        # OpenRouter supports SSE streaming
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        body = json.dumps({
            "model": kwargs.get("model", self.config.model_name),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True
        }).encode()
        
        req = urllib.request.Request(url, data=body, headers={
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        })
        
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as r:
                for line in r:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk['choices'][0]['delta']
                            if 'content' in delta:
                                yield delta['content']
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            yield f"[Error: {e}]"
    
    def _stream_openai(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Streaming from OpenAI."""
        # Similar to OpenRouter
        yield from self._stream_openrouter(prompt, system_prompt, **kwargs)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about current backend."""
        return {
            "backend": self.config.backend.value,
            "mode": self.config.mode.value,
            "model": self.config.model_name or self.config.model_path,
            "context_length": self.config.context_length,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check if LLM is available and working."""
        
        test_prompt = "Trả về JSON: {\"status\": \"ok\"}"
        
        start_time = time.time()
        response = self.generate(test_prompt, max_tokens=50)
        latency = (time.time() - start_time) * 1000
        
        return {
            "healthy": response.error is None,
            "backend": self.config.backend.value,
            "latency_ms": latency,
            "error": response.error
        }
    
    def get_failure_stats(self) -> Dict[str, Any]:
        """Thống kê API failures + fallback."""
        return {
            "backend": self.config.backend.value,
            "api_fail_count": self._api_fail_count,
            "fallback_count": self._fallback_count,
            "fallback_backend": self.fallback_client.config.backend.value if self.fallback_client else None,
            "rate_limiter": _rate_limiter.get_stats(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_dev_client() -> LLMClient:
    """Create LLM client optimized for development."""
    config = LLMConfig(
        mode=LLMMode.DEV,
        max_tokens=256,
        context_length=2048,
        temperature=0.1,
        enable_streaming=False
    )
    return LLMClient(config)


def create_demo_client() -> LLMClient:
    """Create LLM client optimized for demo/production."""
    config = LLMConfig(
        mode=LLMMode.DEMO,
        max_tokens=512,
        context_length=4096,
        temperature=0.1,
        enable_streaming=True
    )
    return LLMClient(config)


def create_client_from_env() -> LLMClient:
    """Create client from environment variables."""
    return LLMClient()
