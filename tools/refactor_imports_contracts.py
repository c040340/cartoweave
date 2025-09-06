import pathlib, re

ROOTS = [pathlib.Path("src"), pathlib.Path("tests"), pathlib.Path("examples")]
PATTERNS = [
    (re.compile(r"\bfrom\s+cartoweave\.compute\s+import\s+SolvePack\b"),
     "from cartoweave.contracts.solvepack import SolvePack"),
    (re.compile(r"\bfrom\s+cartoweave\.compute\.pack\s+import\s+SolvePack\b"),
     "from cartoweave.contracts.solvepack import SolvePack"),
    (re.compile(r"\bfrom\s+cartoweave\.compute\.types\s+import\s+ViewPack\b"),
     "from cartoweave.contracts.viewpack import ViewPack"),
    (re.compile(r"\bfrom\s+cartoweave\.compute\.types\s+import\s+Frame\b"),
     "from cartoweave.contracts.viewpack import Frame"),
]

for root in ROOTS:
    if not root.exists():
        continue
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        new = text
        for pat, repl in PATTERNS:
            new = pat.sub(repl, new)
        if new != text:
            path.write_text(new, encoding="utf-8")
            print("rewrote:", path)
