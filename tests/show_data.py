from pathlib import Path

from npyt import NPYT

path = Path("demo")
files = sorted(path.glob('*.npy'))

for i, f in enumerate(files):
    print(NPYT(f).load(mmap_mode="r+")._raw())
