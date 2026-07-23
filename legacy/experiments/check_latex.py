#!/usr/bin/env python3
"""Check hopf_seam.tex for structural LaTeX issues."""
import re
from collections import Counter

with open('hopf_seam.tex') as f:
    text = f.read()
lines = text.split('\n')

# Check brace balance (ignoring comments and escaped braces)
depth = 0
for i, line in enumerate(lines, 1):
    # Strip comments
    stripped = re.sub(r'(?<!\\)%.*', '', line)
    # Handle escaped braces
    stripped = stripped.replace('\\{', '').replace('\\}', '')
    for ch in stripped:
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
    if depth < 0:
        print(f'Line {i}: brace depth went negative ({depth})')
        depth = 0
if depth != 0:
    print(f'Final brace depth: {depth} (should be 0)')
else:
    print('Brace balance: OK')

# Check begin/end matching
begins = re.findall(r'\\begin\{(\w+)\}', text)
ends = re.findall(r'\\end\{(\w+)\}', text)
bc, ec = Counter(begins), Counter(ends)
for env in sorted(set(list(bc.keys()) + list(ec.keys()))):
    if bc[env] != ec[env]:
        print(f'Environment mismatch: {env} has {bc[env]} begins and {ec[env]} ends')
if bc == ec:
    print('Environment matching: OK')

# Check label/ref consistency
labels = set(re.findall(r'\\label\{([^}]+)\}', text))
refs = set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', text))
undefined = refs - labels
if undefined:
    print(f'Undefined references: {sorted(undefined)}')
else:
    print('All references defined: OK')
unused = labels - refs
if unused:
    print(f'Unused labels (info only): {sorted(unused)}')
