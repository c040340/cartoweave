import pathlib, re

def test_no_engine_or_orchestrators_references():
    root = pathlib.Path("src/cartoweave")
    bad = []
    pat = re.compile(r"cartoweave\.(engine|orchestrators)")
    for p in root.rglob("*.py"):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if pat.search(txt):
            bad.append(str(p))
    assert not bad, "Found forbidden legacy imports:\n" + "\n".join(bad)
