"""Resolve LLM provider keys once, from safe locations only.

Resolution order:
  1. Environment variables (this is what the platform Credentials panel populates).
  2. An operator-managed env file OUTSIDE any git repository, default
     ~/.llm-keys/keys.env, overridable with LLM_KEYS_FILE.

Deliberately NOT supported: reading keys from anywhere inside this repository.
This folder is published to GitHub; a key file here is a leak waiting to happen.

Nothing in this module prints, logs, or returns a key value. `status()` reports
only which providers resolved, so it is safe to call in notebooks and CI logs.

Usage:
    import credentials
    credentials.load()                  # populates os.environ, returns status dict
    credentials.require("anthropic")    # raises with a clear message if absent
"""
import os

# provider -> the environment variable its SDK reads by default
PROVIDERS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "gemini":    "GEMINI_API_KEY",
    "grok":      "XAI_API_KEY",
    "deepseek":  "DEEPSEEK_API_KEY",
    "kimi":      "MOONSHOT_API_KEY",
}

DEFAULT_KEYS_FILE = os.path.expanduser("~/.llm-keys/keys.env")
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parse_env_file(path):
    """Minimal KEY=VALUE parser. Ignores blanks, comments, and `export` prefixes."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load(path=None, verbose=False):
    """Populate os.environ from the key file for any provider not already set."""
    path = path or os.environ.get("LLM_KEYS_FILE") or DEFAULT_KEYS_FILE
    path = os.path.expanduser(path)

    if os.path.abspath(path).startswith(_REPO + os.sep):
        raise RuntimeError(
            f"refusing to read keys from inside the repository ({path}). "
            "This folder is published; keep keys in ~/.llm-keys/keys.env instead.")

    if os.path.exists(path):
        mode = os.stat(path).st_mode & 0o777
        if mode & 0o077:
            print(f"WARNING: {path} is mode {mode:o}; run `chmod 600 {path}`", flush=True)
        for k, v in _parse_env_file(path).items():
            if v and not os.environ.get(k):
                os.environ[k] = v

    st = status()
    if verbose:
        found = [p for p, ok in st.items() if ok]
        print(f"credentials resolved: {found or 'none'}", flush=True)
    return st


def status():
    """Which providers have a non-empty key. Never returns key values."""
    return {p: bool(os.environ.get(var)) for p, var in PROVIDERS.items()}


def require(provider):
    """Return the env var NAME for a provider, after checking it is populated."""
    if provider not in PROVIDERS:
        raise KeyError(f"unknown provider {provider!r}; known: {sorted(PROVIDERS)}")
    var = PROVIDERS[provider]
    if not os.environ.get(var):
        load()
    if not os.environ.get(var):
        raise RuntimeError(
            f"no credential for {provider}. Add it in Customize -> Credentials "
            f"(it will appear as {var}), or put {var}=... in {DEFAULT_KEYS_FILE}.")
    return var


if __name__ == "__main__":
    for p, ok in load(verbose=True).items():
        print(f"  {'yes' if ok else ' no'}  {p:<10} {PROVIDERS[p]}")
