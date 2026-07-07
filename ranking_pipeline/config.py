"""Run-time configuration: the LLM registry, provider selection and API keys.

Everything the pipeline needs to know about a concrete LLM lives in
LLM_REGISTRY -- edit `model` / `api_key_env` there to match what you actually
have access to. The rest of the package only ever reads model settings from
this module.
"""

from pathlib import Path

from dotenv import load_dotenv

# Repository root = the folder containing this package.
ROOT_DIR = Path(__file__).resolve().parents[1]

# API keys live in an env file at the repository root; anchoring the path to
# __file__ means loading works no matter what the working directory is
# (load_dotenv is a silent no-op when the file is missing).
ENV_FILE = ROOT_DIR / "API_keys.env"


def load_api_keys() -> None:
    """Load provider API keys from API_keys.env into the environment."""
    load_dotenv(ENV_FILE)


# Local Qwen3 thinking mode. True = Qwen reasons before answering (slower;
# bump --llm-max-tokens to ~2048+ so the JSON isn't truncated). Only applied
# to the openai_compat (local vLLM) provider; ignored for API models.
# Toggling it partitions the LLM cache, so thinking and non-thinking results
# never mix. Flipped at runtime by llm_ranking_pipeline.py's --qwen-thinking
# flag; the default here is what you get without the flag.
QWEN_ENABLE_THINKING = False


# ---------------------------------------------------------------------------
# LLM registry
# ---------------------------------------------------------------------------
# `provider`:
#   openai_compat -> OpenAI-compatible HTTP endpoint (local vLLM); api key optional
#   openai        -> OpenAI API (needs api_key_env)
#   anthropic     -> Anthropic API (needs api_key_env)
# `max_concurrent` -> cap on in-flight requests.
# `rate_limit`     -> max request *starts* per second (None = uncapped). Use this
#                     to stay under provider RPM/TPS limits. Local vLLM has no
#                     such limit, so leave it None there -- it is bounded only by
#                     max_concurrent and the on-disk cache. Tune the API values
#                     to your account tier.
#
# NOTE: the API model id strings below are best-effort defaults. Verify them
# against your provider's current model list before a real run.
LLM_REGISTRY: dict[str, dict] = {
    "qwen-local": {
        "provider": "openai_compat",
        "model": "Qwen/Qwen3-14B-AWQ",
        "base_url": "http://localhost:8000/v1",
        "api_key_env": None,
        "max_concurrent": 99,
        "rate_limit": None,               # local: no per-second cap
    },
    "gpt-5.4-nano": {
        "provider": "openai",
        "model": "gpt-5.4-nano",          # verify exact id
        "base_url": None,                 # default OpenAI endpoint
        "api_key_env": "OPENAI_API_KEY",
        "max_concurrent": 8,
        "rate_limit": 8,                  # req/s -- tune to your tier
    },
    "gpt-5.4-mini": {
        "provider": "openai",
        "model": "gpt-5.4-mini",          # verify exact id
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "max_concurrent": 8,
        "rate_limit": 8,
    },
    "claude-haiku": {
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20251001",   # verify exact id
        "base_url": None,
        "api_key_env": "ANTHROPIC_API_KEY",
        "max_concurrent": 8,
        "rate_limit": 8,
    },
}

LOCAL_PROVIDERS = {"openai_compat"}


def resolve_llms(selection: list[str]) -> list[str]:
    """Map a --llms selection ('local', 'all', or explicit keys) to registry keys."""
    if selection == ["all"]:
        return list(LLM_REGISTRY)
    if selection == ["local"]:
        return [k for k, v in LLM_REGISTRY.items() if v["provider"] in LOCAL_PROVIDERS]
    bad = [k for k in selection if k not in LLM_REGISTRY]
    if bad:
        raise SystemExit(
            f"Unknown --llms entries {bad}. "
            f"Use 'local', 'all', or keys: {list(LLM_REGISTRY)}"
        )
    return selection
