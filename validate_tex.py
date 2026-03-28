import re
from collections import Counter

with open("hopf_seam.tex") as f:
    tex = f.read()

begins = re.findall(r"\\begin\{(\w+)\}", tex)
ends = re.findall(r"\\end\{(\w+)\}", tex)
print(f"begin: {len(begins)}, end: {len(ends)}")

bc, ec = Counter(begins), Counter(ends)
for env in sorted(set(list(bc.keys()) + list(ec.keys()))):
    b, e = bc.get(env, 0), ec.get(env, 0)
    if b != e:
        print(f"  MISMATCH {env}: {b} vs {e}")

labels = re.findall(r"\\label\{([^}]+)\}", tex)
lc = Counter(labels)
dups = {k: v for k, v in lc.items() if v > 1}
print(f"Labels: {len(labels)}, dups: {dups if dups else 'none'}")

for lbl in ["lem:riemann-expansion", "eq:R-expansion", "eq:R2-christoffel"]:
    print(f"  {lbl}: {'found' if lbl in labels else 'MISSING'}")

refs = re.findall(r"\\ref\{([^}]+)\}", tex)
print(f"  refs to lem:riemann-expansion: {refs.count('lem:riemann-expansion')}")

# Check for unresolved refs
all_labels = set(labels)
all_refs = set(refs)
unresolved = all_refs - all_labels
if unresolved:
    print(f"UNRESOLVED REFS: {unresolved}")
else:
    print("All refs resolved")
