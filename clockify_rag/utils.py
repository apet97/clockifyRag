"""Utility functions for file I/O, validation, logging, and text processing."""

import atexit
import hashlib
import json
import logging
import os
import platform
import pathlib
import re
import tempfile
import time
import unicodedata
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import requests

# Note: config will be imported at module level to avoid circular imports
# For functions that need dynamic config access, we'll import inside functions

logger = logging.getLogger(__name__)

ALLOWED_CORPUS_FILENAME = "knowledge_helpcenter.md"
DEFAULT_CORPUS_FILES = [ALLOWED_CORPUS_FILENAME]


def resolve_corpus_path(preferred: Optional[str] = None) -> tuple[str, bool, List[str]]:
    """Return a usable corpus path for ingestion (knowledge_helpcenter.md only).

    Args:
        preferred: Explicit path provided by the caller (highest priority).

    Returns:
        (path, exists, candidates_checked)
    """
    candidates: List[str] = []
    if preferred:
        preferred = os.path.expandvars(os.path.expanduser(preferred))
        candidates.append(preferred)
    candidates.extend(DEFAULT_CORPUS_FILES)
    package_root = pathlib.Path(__file__).resolve().parents[1]
    candidates.append(str(package_root / ALLOWED_CORPUS_FILENAME))

    # Deduplicate while preserving order
    seen: List[str] = []
    for cand in candidates:
        if cand not in seen:
            seen.append(cand)

    if preferred and pathlib.Path(preferred).name != ALLOWED_CORPUS_FILENAME:
        return preferred, False, [ALLOWED_CORPUS_FILENAME]

    for cand in seen:
        if os.path.isfile(cand):
            return cand, True, seen

    if preferred:
        return preferred, False, seen

    # Fall back to the only supported corpus name for a clear error
    return ALLOWED_CORPUS_FILENAME, False, [ALLOWED_CORPUS_FILENAME]


# ====== INPUT SANITIZATION ======
def sanitize_for_log(text: str, max_length: int = 1000) -> str:
    """Sanitize text for safe logging to prevent log injection attacks.

    FIX (Error #6): Removes control characters and truncates to prevent log injection.

    Args:
        text: Text to sanitize
        max_length: Maximum length after sanitization

    Returns:
        Sanitized text safe for logging
    """
    # Remove control characters except tab
    sanitized = "".join(ch if ch.isprintable() or ch in ("\t",) else f"\\x{ord(ch):02x}" for ch in text)

    # Truncate to max length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."

    return sanitized


# ====== CLEANUP HANDLERS ======
def _release_lock_if_owner():
    """Release build lock on exit if held by this process."""
    from .config import BUILD_LOCK

    try:
        if os.path.exists(BUILD_LOCK):
            with open(BUILD_LOCK) as f:
                data = json.loads(f.read())
            if data.get("pid") == os.getpid():
                os.remove(BUILD_LOCK)
                logger.debug("Cleaned up build lock")
    except (OSError, FileNotFoundError, json.JSONDecodeError, KeyError):
        # Cleanup failed - not critical, can ignore
        pass


atexit.register(_release_lock_if_owner)


# ====== PROCESS & LOCK MANAGEMENT ======
def _pid_alive(pid: int) -> bool:
    """Check if a process is alive. Cross-platform."""
    if pid <= 0:
        return False
    system = platform.system().lower()
    try:
        if system != "windows":
            # POSIX: use signal 0 check
            os.kill(pid, 0)
            return True
        else:
            # Windows: best-effort with optional psutil
            try:
                import psutil

                return psutil.pid_exists(pid)
            except Exception:
                # Fallback: treat as alive; bounded wait handles stale locks
                # Hint once for better DX
                try:
                    if not getattr(_pid_alive, "_hinted_psutil", False):
                        logger.debug(
                            "[build_lock] psutil not available on Windows; install 'psutil' for precise PID checks"
                        )
                        _pid_alive._hinted_psutil = True  # type: ignore
                except Exception:
                    pass
                return True
    except OSError:
        return False


@contextmanager
def build_lock():
    """Exclusive build lock with atomic create (O_EXCL) and stale-lock recovery.

    Uses atomic file creation to prevent partial writes. Detects stale locks via
    PID liveness check and TTL expiration.
    """
    from .config import BUILD_LOCK, BUILD_LOCK_TTL_SEC

    pid = os.getpid()
    hostname = platform.node() or "unknown"
    deadline = time.time() + 10.0  # 10s max wait

    while True:
        try:
            # Atomic create: fails if file exists (O_EXCL)
            fd = os.open(BUILD_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                with os.fdopen(fd, "w") as f:
                    started_at = time.time()
                    lock_data = {
                        "pid": pid,
                        "host": hostname,
                        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
                        "started_at_epoch": started_at,
                        "ttl_sec": BUILD_LOCK_TTL_SEC,
                    }
                    f.write(json.dumps(lock_data))
                break  # Successfully acquired lock
            except Exception:
                try:
                    os.close(fd)
                except Exception:
                    pass
                raise
        except FileExistsError:
            # Lock file exists; check if it's stale
            try:
                with open(BUILD_LOCK, "r") as f:
                    lock_data = json.loads(f.read())
                stale_pid = lock_data.get("pid", 0)
                started_at_epoch = lock_data.get("started_at_epoch", 0)
                ttl_sec = lock_data.get("ttl_sec", BUILD_LOCK_TTL_SEC)

                # Check TTL expiration
                age = time.time() - started_at_epoch
                is_expired = age > ttl_sec
                pid_alive = _pid_alive(stale_pid)

                # If expired or dead owner, try to remove and retry
                if is_expired or not pid_alive:
                    reason = f"expired (age={age:.1f}s > ttl={ttl_sec}s)" if is_expired else f"dead PID {stale_pid}"
                    logger.warning(f"[build_lock] Recovering: {reason}")
                    try:
                        os.remove(BUILD_LOCK)
                        continue  # Retry atomic create
                    except Exception:
                        pass
            except FileNotFoundError:
                # Lock removed by another process between check and read, retry
                logger.debug("[build_lock] Lock removed during check, retrying...")
                continue
            except (json.JSONDecodeError, ValueError) as e:
                # Corrupt lock file, try to remove and retry
                logger.warning(f"[build_lock] Corrupt lock file: {e}")
                try:
                    os.remove(BUILD_LOCK)
                except Exception:
                    pass
                continue
            except Exception as e:
                logger.warning(f"[build_lock] Unexpected error reading lock: {e}")
                # Fall through to timeout logic

            # Still held by live process; wait and retry with 250 ms polling
            if time.time() > deadline:
                raise RuntimeError("Build already in progress; timed out waiting for lock release")
            while time.time() < deadline:  # Use deadline directly
                time.sleep(0.25)
                if not os.path.exists(BUILD_LOCK):
                    break
                if time.time() > deadline:  # Check deadline in loop
                    raise RuntimeError("Build already in progress; timed out waiting for lock release")
            continue

    try:
        yield
    finally:
        # Only remove lock if we still own it
        try:
            with open(BUILD_LOCK, "r") as f:
                lock_data = json.loads(f.read())
            if lock_data.get("pid") == os.getpid():
                os.remove(BUILD_LOCK)
        except Exception as e:
            # Non-critical: lock file may already be removed or unreadable
            logger.debug("Failed to clean up build lock: %s", e)


# ====== CONFIG VALIDATION ======
def validate_ollama_url(url: str) -> str:
    """Validate and normalize Ollama URL. Returns validated URL."""
    from urllib.parse import urlparse

    try:
        candidate = url.strip()
        if "://" not in candidate:
            candidate = "http://" + candidate

        parsed = urlparse(candidate)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid scheme: {parsed.scheme}. Must be http or https.")
        if not parsed.netloc:
            raise ValueError(f"Invalid URL: {candidate}. Must include host.")
        # Normalize: ensure no trailing slash
        url = f"{parsed.scheme}://{parsed.netloc}"
        if parsed.path and parsed.path != "/":
            url += parsed.path
        return url
    except Exception as e:
        raise ValueError(f"Invalid Ollama URL '{url}': {e}")


def check_ollama_connectivity(url: str, timeout: float = 3.0) -> str:
    """Validate the URL and perform a lightweight connectivity probe.

    Args:
        url: Ollama base URL or host[:port]
        timeout: Request timeout in seconds

    Returns:
        Normalized URL (scheme + host[:port])

    Raises:
        RuntimeError: if the probe fails
    """
    normalized = validate_ollama_url(url)
    probe_url = f"{normalized}/api/tags"
    try:
        from .config import allow_proxies_enabled

        proxies = None
        if not allow_proxies_enabled():
            proxies = {"http": None, "https": None}

        response = requests.get(probe_url, timeout=timeout, allow_redirects=False, proxies=proxies)
        response.raise_for_status()
        logger.debug("Ollama connectivity OK: %s", probe_url)
        return normalized
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Failed to connect to Ollama at {normalized}: {exc}") from exc


def validate_and_set_config(
    ollama_url=None,
    gen_model=None,
    emb_model=None,
    ctx_budget=None,
    emb_backend=None,
    ann_backend=None,
    alpha_hybrid=None,
    top_k=None,
    pack_top=None,
    threshold=None,
    seed=None,
    num_ctx=None,
    num_predict=None,
    retries=None,
    faiss_multiplier=None,
):
    """Validate and set global config from CLI args."""
    import clockify_rag.config as config

    if ollama_url:
        config.RAG_OLLAMA_URL = validate_ollama_url(ollama_url)
        config.OLLAMA_URL = config.RAG_OLLAMA_URL  # legacy alias
        logger.info(f"Ollama endpoint: {config.RAG_OLLAMA_URL}")

    if gen_model:
        config.RAG_CHAT_MODEL = gen_model
        config.GEN_MODEL = config.RAG_CHAT_MODEL  # legacy alias
        logger.info(f"Generation model: {config.RAG_CHAT_MODEL}")

    if emb_model:
        config.RAG_EMBED_MODEL = emb_model
        config.EMB_MODEL = config.RAG_EMBED_MODEL  # legacy alias
        logger.info(f"Embedding model: {config.RAG_EMBED_MODEL}")

    if emb_backend:
        config.EMB_BACKEND = emb_backend
        logger.info(f"Embedding backend: {config.EMB_BACKEND}")

    if ann_backend:
        config.USE_ANN = ann_backend
        logger.info(f"ANN backend: {config.USE_ANN}")

    if alpha_hybrid is not None:
        config.ALPHA_HYBRID = alpha_hybrid
        logger.info(f"Alpha hybrid: {config.ALPHA_HYBRID}")

    if ctx_budget:
        try:
            config.CTX_TOKEN_BUDGET = int(ctx_budget)
            if config.CTX_TOKEN_BUDGET < 256:
                raise ValueError("Context budget must be >= 256")
            logger.info(f"Context token budget: {config.CTX_TOKEN_BUDGET}")
        except ValueError as e:
            raise ValueError(f"Invalid context budget: {e}")

    if top_k:
        config.DEFAULT_TOP_K = top_k
        logger.info(f"Default top k: {config.DEFAULT_TOP_K}")

    if pack_top:
        config.DEFAULT_PACK_TOP = pack_top
        logger.info(f"Default pack top: {config.DEFAULT_PACK_TOP}")

    if threshold is not None:
        config.DEFAULT_THRESHOLD = threshold
        logger.info(f"Default threshold: {config.DEFAULT_THRESHOLD}")

    if seed is not None:
        config.DEFAULT_SEED = seed
        logger.info(f"Default seed: {config.DEFAULT_SEED}")

    if num_ctx:
        config.DEFAULT_NUM_CTX = num_ctx
        logger.info(f"Default num ctx: {config.DEFAULT_NUM_CTX}")

    if num_predict:
        config.DEFAULT_NUM_PREDICT = num_predict
        logger.info(f"Default num predict: {config.DEFAULT_NUM_PREDICT}")

    if retries is not None:
        config.DEFAULT_RETRIES = retries
        logger.info(f"Default retries: {config.DEFAULT_RETRIES}")

    if faiss_multiplier:
        config.FAISS_CANDIDATE_MULTIPLIER = faiss_multiplier
        logger.info(f"FAISS candidate multiplier: {config.FAISS_CANDIDATE_MULTIPLIER}")

    # Recompute derived globals so downstream code sees consistent dims/models
    if hasattr(config, "refresh_runtime_settings"):
        config.refresh_runtime_settings()


def validate_chunk_config():
    """Validate chunk parameters at startup."""
    from .config import CHUNK_CHARS, CHUNK_OVERLAP

    if CHUNK_OVERLAP >= CHUNK_CHARS:
        raise ValueError(f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be < CHUNK_CHARS ({CHUNK_CHARS})")
    logger.debug(f"Chunk config: size={CHUNK_CHARS}, overlap={CHUNK_OVERLAP}")


def check_pytorch_mps():
    """Check PyTorch MPS availability on M1 Macs and log warnings (v4.1.2)."""
    is_macos_arm64 = platform.system() == "Darwin" and platform.machine() == "arm64"

    if not is_macos_arm64:
        return  # Only relevant for M1/M2/M3 Macs

    try:
        import torch

        mps_available = torch.backends.mps.is_available()

        if mps_available:
            logger.info("info: pytorch_mps=available platform=arm64 (GPU acceleration enabled)")
        else:
            logger.warning(
                "warning: pytorch_mps=unavailable platform=arm64 "
                "hint='Embeddings will use CPU (slower). Ensure macOS 12.3+ and PyTorch 1.12+'"
            )
            logger.warning("  To fix: pip install --upgrade torch or conda install -c pytorch pytorch")
    except ImportError:
        logger.debug("info: pytorch not imported, skipping MPS check")
    except Exception as e:
        logger.debug(f"info: pytorch_mps check failed: {e}")


def _log_config_summary(
    use_rerank=False, pack_top=None, seed=None, threshold=None, top_k=None, num_ctx=None, num_predict=None, retries=0
):
    """Log configuration summary at startup."""
    from .config import (
        GEN_MODEL,
        EMB_MODEL,
        DEFAULT_PACK_TOP,
        DEFAULT_SEED,
        DEFAULT_THRESHOLD,
        DEFAULT_TOP_K,
        DEFAULT_NUM_CTX,
        DEFAULT_NUM_PREDICT,
        EMB_READ_T,
        CHAT_READ_T,
        RERANK_READ_T,
        REFUSAL_STR,
    )

    # Use defaults if not provided
    pack_top = pack_top or DEFAULT_PACK_TOP
    seed = seed or DEFAULT_SEED
    threshold = threshold or DEFAULT_THRESHOLD
    top_k = top_k or DEFAULT_TOP_K
    num_ctx = num_ctx or DEFAULT_NUM_CTX
    num_predict = num_predict or DEFAULT_NUM_PREDICT

    # IMPROVEMENT: Log platform and architecture for M1 compatibility verification
    # See: CODEBASE_HEALTH_REVIEW_2025-11-09.md - Apple Silicon (M1/M2/M3) Compatibility
    import platform

    sys_platform = platform.system()
    sys_arch = platform.machine()
    is_macos_arm64 = sys_platform == "Darwin" and sys_arch == "arm64"

    if is_macos_arm64:
        logger.info(
            f"PLATFORM platform={sys_platform} arch={sys_arch} (Apple Silicon detected - using M1-optimized settings)"
        )
    else:
        logger.info(f"PLATFORM platform={sys_platform} arch={sys_arch}")

    from .config import ALLOW_PROXIES

    proxy_trust = 1 if ALLOW_PROXIES else 0
    # Single-line CONFIG banner
    logger.info(
        f"CONFIG model={GEN_MODEL} emb={EMB_MODEL} topk={top_k} pack={pack_top} thr={threshold} "
        f"seed={seed} ctx={num_ctx} pred={num_predict} retries={retries} "
        f"timeouts=(3,{int(EMB_READ_T)}/{int(CHAT_READ_T)}/{int(RERANK_READ_T)}) "
        f"trust_env={proxy_trust} rerank={1 if use_rerank else 0}"
    )
    # Print refusal string once for sanity
    logger.info(f'REFUSAL_STR="{REFUSAL_STR}"')


# ====== FILE I/O UTILITIES ======
def _fsync_dir(path: str) -> None:
    """Sync directory to ensure durability (best-effort, platform-dependent)."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    try:
        fd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception as e:
        # Best-effort on platforms/filesystems without dir fsync
        logger.debug("Directory fsync skipped for %s: %s", d, e)


def atomic_write_bytes(path: str, data: bytes) -> None:
    """Atomically write bytes with fsync durability."""
    tmp = None
    try:
        d = os.path.dirname(os.path.abspath(path)) or "."
        with tempfile.NamedTemporaryFile(prefix=".tmp.", dir=d, delete=False) as f:
            tmp = f.name
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(path)
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def atomic_write_text(path: str, text: str) -> None:
    """Atomically write text file with fsync durability."""
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str, obj: Any) -> None:
    """Atomically write JSON file."""
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False))


def atomic_write_jsonl(path: str, rows_list: list) -> None:
    """Atomically write JSONL file (list of dicts)."""
    # Build rows in memory as list of JSON strings
    lines = []
    for row in rows_list:
        if isinstance(row, dict):
            lines.append(json.dumps(row, ensure_ascii=False))
        else:
            lines.append(str(row))
    content = "\n".join(lines)
    if content and not content.endswith("\n"):
        content += "\n"
    atomic_write_text(path, content)


def atomic_save_npy(arr: np.ndarray, path: str) -> None:
    """Atomically save numpy array with fsync durability. Enforce float32."""
    # Enforce float32
    arr = arr.astype("float32")
    d = os.path.dirname(os.path.abspath(path)) or "."
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".tmp.", dir=d, delete=False) as f:
            tmp = f.name
            np.save(f, arr)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(path)
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception as e:
                logger.debug("Failed to clean up temp file %s: %s", tmp, e)


# ====== LOGGING UTILITIES ======
def log_event(event: str, **fields):
    """Log a structured JSON event. Fallback to plain format if JSON serialization fails."""
    try:
        record = {"event": event, **fields}
        logger.info(json.dumps(record, ensure_ascii=False))
    except Exception as e:
        # Fallback to plain string if JSON encoding fails
        logger.debug("JSON serialization failed for event %s: %s", event, e)
        logger.info(f"{event} {fields}")


def log_kpi(topk: int, packed: int, used_tokens: int, rerank_applied: bool, rerank_reason: str = ""):
    """Log KPI metrics."""
    from .config import KPI

    logger.info(
        f"KPI retrieve={KPI.retrieve_ms}ms ann={KPI.ann_ms}ms "
        f"rerank={KPI.rerank_ms}ms ask={KPI.ask_ms}ms "
        f"topk={topk} packed={packed} used_tokens={used_tokens} "
        f"rerank={1 if rerank_applied else 0} reason={rerank_reason}"
    )


# ====== TEXT PROCESSING UTILITIES ======
def norm_ws(s: str) -> str:
    """Normalize whitespace."""
    return re.sub(r"[ \t]+", " ", s.strip())


def is_rtf(text: str) -> bool:
    """Check if text is RTF format."""
    # Check first 128 chars for RTF signature
    head_128 = text[:128]
    if "{\\rtf" in head_128 or "\\rtf" in head_128:
        return True

    # Check first 4096 chars for RTF control words (stricter)
    head_4k = text[:4096]
    rtf_commands = re.findall(r"\\(?:cf\d+|u[+-]?\d+\?|f\d+|pard)\b", head_4k)
    return len(rtf_commands) > 20


def strip_noise(text: str) -> str:
    """Drop scrape artifacts and normalize encoding."""
    # Guard: only apply RTF stripping if content is likely RTF
    if is_rtf(text):
        # Strip RTF escapes only for RTF content (more precise patterns)
        text = re.sub(r"\\cf\d+", "", text)  # \cfN (color)
        text = re.sub(r"\\u[+-]?\d+\?", "", text)  # \u1234? (unicode)
        text = re.sub(r"\\f\d+", "", text)  # \fN (font)
        text = re.sub(r"\{\\\*[^}]*\}", "", text)  # {\* ... } (special)
        text = re.sub(r"\\pard\b[^\n]*", "", text)  # \pard (paragraph)
    # Always remove chunk markers
    text = re.sub(r"^## +Chunk +\d+\s*$", "", text, flags=re.M)
    return text


def tokenize(s: str) -> list:
    """Simple tokenizer: lowercase [a-z0-9]+."""
    s = s.lower()
    s = unicodedata.normalize("NFKC", s)
    return re.findall(r"[a-z0-9]+", s)


def approx_tokens(chars: int) -> int:
    """Estimate tokens: 1 token ≈ 4 chars."""
    return max(1, chars // 4)


def compute_sha256(filepath: str) -> str:
    """Compute SHA256 hash of a file or directory (stable order)."""
    sha256 = hashlib.sha256()
    path_obj = pathlib.Path(filepath)

    if path_obj.is_dir():
        # Hash relative paths + contents of all files to detect KB changes
        files = sorted(p for p in path_obj.rglob("*") if p.is_file())
        for file_path in files:
            rel = file_path.relative_to(path_obj).as_posix()
            sha256.update(rel.encode("utf-8"))
            with open(file_path, "rb") as f:
                while True:
                    data = f.read(65536)
                    if not data:
                        break
                    sha256.update(data)
        return sha256.hexdigest()

    with open(filepath, "rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def truncate_to_token_budget(text: str, budget: int) -> str:
    """Truncate text to fit token budget, append ellipsis."""
    est_tokens = approx_tokens(len(text))
    if est_tokens <= budget:
        return text
    # Approximate char count for budget
    target_chars = budget * 4
    if len(text) <= target_chars:
        return text
    return text[:target_chars] + " […]"


# ====== NLTK INITIALIZATION ======
_NLTK_AVAILABLE = False
_NLTK_DOWNLOAD_ATTEMPTED = False


def _ensure_nltk(auto_download=None):
    """Ensure NLTK is available, with optional download control for offline environments."""
    global _NLTK_AVAILABLE, _NLTK_DOWNLOAD_ATTEMPTED

    if _NLTK_AVAILABLE:
        return True

    try:
        import nltk
    except ImportError:
        logger.warning("NLTK not installed. Chunking will use simpler fallback.")
        return False

    # Check if we already have punkt
    try:
        nltk.data.find("tokenizers/punkt")
        _NLTK_AVAILABLE = True
        return True
    except LookupError:
        pass

    # Determine if we should download
    import clockify_rag.config as config

    if auto_download is None:
        # Use centralized config (default: allow download unless explicitly disabled)
        auto_download = config.NLTK_AUTO_DOWNLOAD

    if auto_download and not _NLTK_DOWNLOAD_ATTEMPTED:
        _NLTK_DOWNLOAD_ATTEMPTED = True
        logger.info("Downloading NLTK punkt tokenizer (one-time setup)...")
        try:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)  # For newer NLTK versions
            _NLTK_AVAILABLE = True
            logger.info("NLTK punkt downloaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"NLTK download failed ({e}). Using simpler chunking fallback.")
            return False
    else:
        if not auto_download:
            logger.warning("NLTK auto-download disabled (NLTK_AUTO_DOWNLOAD=0). Using simpler chunking fallback.")
        return False


# ====== LOCAL EMBEDDINGS ======
_ST_ENCODER = None
_ST_BATCH_SIZE = 96


def _load_st_encoder():
    """Lazy-load SentenceTransformer model once."""
    global _ST_ENCODER
    if _ST_ENCODER is None:
        from sentence_transformers import SentenceTransformer

        _ST_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
        logger.debug("Loaded SentenceTransformer: all-MiniLM-L6-v2 (384-dim)")
    return _ST_ENCODER


# ====== FAISS INDEX ======
def _try_load_faiss():
    """Try importing FAISS; returns None if not available."""
    try:
        import faiss

        return faiss
    except ImportError:
        logger.info("info: ann=fallback reason=missing-faiss")
        return None


# ====== POLICY GUARDRAILS ======
def looks_sensitive(question: str) -> bool:
    """Check if question involves sensitive intent (account/billing/PII)."""
    sensitive_keywords = {
        # Financial
        "invoice",
        "billing",
        "credit card",
        "payment",
        "salary",
        "account balance",
        # Authentication & Secrets
        "password",
        "token",
        "api key",
        "secret",
        "private key",
        # PII
        "ssn",
        "social security",
        "iban",
        "swift",
        "routing number",
        "account number",
        "phone number",
        "email address",
        "home address",
        "date of birth",
        # Compliance
        "gdpr",
        "pii",
        "personally identifiable",
        "personal data",
    }
    q_lower = question.lower()
    return any(kw in q_lower for kw in sensitive_keywords)


def inject_policy_preamble(snippets_block: str, question: str) -> str:
    """Optionally prepend policy reminder for sensitive queries."""
    if looks_sensitive(question):
        policy = "[INTERNAL POLICY]\nDo not reveal PII, account secrets, or payment details. For account changes, redirect to secure internal admin panel.\n\n"
        return policy + snippets_block
    return snippets_block


# ====== INPUT SANITIZATION ======
def sanitize_question(q: str, max_length: int = 2000) -> str:
    """Validate and sanitize user question.

    Args:
        q: User question string
        max_length: Maximum allowed question length (default: 2000)

    Returns:
        Sanitized question string

    Raises:
        ValueError: If question is invalid (empty, too long, invalid characters)
    """
    # Type check
    if not isinstance(q, str):
        raise ValueError(f"Question must be a string, got {type(q).__name__}")

    # Strip whitespace
    q = q.strip()

    # Check length
    if len(q) == 0:
        raise ValueError("Question cannot be empty. Hint: Provide a meaningful question about Clockify.")
    if len(q) > max_length:
        raise ValueError(
            f"Question too long (max {max_length} characters, got {len(q)}).\n"
            f"Hint: Break your question into smaller, focused queries."
        )

    # Check for null bytes first (specific check)
    if "\x00" in q:
        raise ValueError("Question contains control characters")

    # Check for control characters (except newline, tab, carriage return)
    if any(ord(c) < 32 and c not in "\n\r\t" for c in q):
        raise ValueError("Question contains invalid control characters")

    # Check for suspicious patterns (basic prompt injection detection)
    suspicious_patterns = [
        "<script",
        "javascript:",
        "eval(",
        "exec(",
        "__import__",
        "<?php",
    ]
    q_lower = q.lower()
    for pattern in suspicious_patterns:
        if pattern in q_lower:
            raise ValueError(f"Question contains suspicious pattern: {pattern}")

    return q


# ====== ENHANCED STRUCTURED LOGGING (Analysis Section 10.1) ======
def log_query_metrics(
    question: str,
    answer: str,
    confidence: Optional[int],
    timing: Dict[str, float],
    metadata: Dict[str, Any],
    routing: Optional[Dict[str, Any]] = None,
) -> None:
    """Log structured query metrics for monitoring dashboard.

    OPTIMIZATION (Analysis Section 10.1): Enhanced structured logging for monitoring.
    Provides comprehensive metrics in JSON format for dashboard integration.

    Args:
        question: User question
        answer: Generated answer
        confidence: Confidence score (0-100) or None
        timing: Timing breakdown dict
        metadata: Query metadata
        routing: Confidence routing metadata (optional)
    """
    import datetime

    # Extract routing info if available
    routing_action = "unknown"
    routing_level = "unknown"
    escalated = False
    if routing:
        routing_action = routing.get("action", "unknown")
        routing_level = routing.get("level", "unknown")
        escalated = routing.get("escalated", False)

    from .config import REFUSAL_STR

    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "event": "query_processed",
        "question_length": len(question),
        "question_preview": question[:100],
        "answer_length": len(answer),
        "refused": (answer == REFUSAL_STR),
        "confidence": confidence,
        "timing_ms": {
            "total": timing.get("total_ms", 0),
            "retrieve": timing.get("retrieve_ms", 0),
            "mmr": timing.get("mmr_ms", 0),
            "rerank": timing.get("rerank_ms", 0),
            "llm": timing.get("llm_ms", 0),
        },
        "retrieval": {
            "count": metadata.get("retrieval_count", 0),
            "packed": metadata.get("packed_count", 0),
            "tokens_used": metadata.get("used_tokens", 0),
            "rerank_applied": metadata.get("rerank_applied", False),
        },
        "routing": {
            "action": routing_action,
            "level": routing_level,
            "escalated": escalated,
        },
    }

    logger.info(json.dumps(log_entry))


def log_performance_metrics(
    operation: str, duration_ms: float, success: bool, metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log performance metrics for operations.

    Args:
        operation: Operation name (e.g., "build_index", "retrieve", "embed")
        duration_ms: Operation duration in milliseconds
        success: Whether operation succeeded
        metadata: Additional metadata (optional)
    """
    import datetime

    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "event": "performance_metric",
        "operation": operation,
        "duration_ms": duration_ms,
        "success": success,
    }

    if metadata:
        log_entry["metadata"] = metadata

    logger.info(json.dumps(log_entry))
