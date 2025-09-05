import re
import pathlib


def test_no_tiny_literals_in_compute():
    root = pathlib.Path("src/cartoweave/compute")
    bad = []
    pat = re.compile(r"(?<!\w)(?:1e-1[0-9]|1e-9|1e-10|1e-11|1e-12)(?!\w)", re.IGNORECASE)
    for p in root.rglob("*.py"):
        if "tests" in p.as_posix():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(txt.splitlines(), start=1):
            if pat.search(line):
                bad.append(f"{p}:{i}:{line.strip()}")
    assert not bad, "Found hard-coded tiny eps literals:\n" + "\n".join(bad[:20])

