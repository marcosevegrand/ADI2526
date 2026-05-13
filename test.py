import sys
lines = open('docs/relatorio/ADI 25_26.md').read().split('\n')
for i, l in enumerate(lines):
    if l.strip():
        print(f"{i}: {l[:50]}")
