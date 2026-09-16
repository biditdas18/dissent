"""Deterministic, resource-bounded execution of model-generated Python.

This is the *symbolic* half of the study: it turns a generated program into a
checkable consequence. It is a correctness/robustness sandbox for benign
benchmark code, not a security boundary against adversarial code.
"""
import subprocess, sys, tempfile, os, re, json, hashlib

TIMEOUT_S = 8
MAX_OUTPUT = 4000

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)


def extract_code(text):
    """Last fenced python block; else None. Deterministic."""
    if not text:
        return None
    blocks = _FENCE.findall(text)
    if not blocks:
        return None
    return blocks[-1].strip() or None


_PRELUDE = (
    "import resource as _r, sys as _s\n"
    "def _cap(which, want):\n"
    "    try:\n"
    "        soft, hard = _r.getrlimit(which)\n"
    "        if hard != _r.RLIM_INFINITY:\n"
    "            want = min(want, hard)\n"
    "        _r.setrlimit(which, (want, hard))\n"
    "    except Exception:\n"
    "        pass\n"
    "_cap(_r.RLIMIT_AS, 2 * 1024**3)\n"
    "_cap(_r.RLIMIT_CPU, 8)\n"
    "_s.setrecursionlimit(10000)\n"
    "del _cap\n"
)


def run_code(code, timeout=TIMEOUT_S):
    """Execute `code` in a fresh interpreter. Returns a JSON-safe dict."""
    if not code:
        return {"ok": False, "stdout": "", "stderr": "", "error": "no_code",
                "returncode": None, "code_sha": None}
    sha = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "prog.py")
        with open(path, "w") as f:
            f.write(_PRELUDE + code)
        env = {"PATH": "/usr/bin:/bin", "HOME": td, "TMPDIR": td,
               "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            p = subprocess.run([sys.executable, "-I", "-S", path], cwd=td, env=env,
                               capture_output=True, text=True, timeout=timeout)
            return {"ok": p.returncode == 0,
                    "stdout": (p.stdout or "")[:MAX_OUTPUT],
                    "stderr": (p.stderr or "")[-1500:],
                    "error": None if p.returncode == 0 else "nonzero_exit",
                    "returncode": p.returncode, "code_sha": sha}
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "stderr": "", "error": "timeout",
                    "returncode": None, "code_sha": sha}
        except Exception as e:
            return {"ok": False, "stdout": "", "stderr": "", "error": repr(e)[:200],
                    "returncode": None, "code_sha": sha}


_NUM = re.compile(r"-?[\d,]+(?:\.\d+)?")


def last_number(s):
    """Last numeric token in stdout, normalised. None if absent."""
    if not s:
        return None
    m = _NUM.findall(s.strip())
    if not m:
        return None
    v = m[-1].replace(",", "")
    try:
        f = float(v)
    except ValueError:
        return None
    return str(int(f)) if f == int(f) else str(f)


def run_and_value(text_or_code, is_code=False, timeout=TIMEOUT_S):
    """Extract (if needed), execute, and read off the printed value."""
    code = text_or_code if is_code else extract_code(text_or_code)
    r = run_code(code, timeout=timeout)
    r["value"] = last_number(r["stdout"]) if r["ok"] else None
    r["code"] = code
    return r


if __name__ == "__main__":
    demo = "x = 5 * 7\nprint(x - 3)\n"
    print(json.dumps(run_and_value(demo, is_code=True), indent=1))
    print(json.dumps(run_and_value("```python\nprint(1/0)\n```"), indent=1))
    print(json.dumps(run_and_value("no code here"), indent=1))
