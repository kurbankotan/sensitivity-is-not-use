"""Patent basvurusu yapildiktan sonra calistirilir.

Depodaki gizlilik uyarilarini kaldirir; baska hicbir seye dokunmaz.
Once --dry-run ile calistirip neyin degisecegini gorun.

    python tools/make_public.py --dry-run
    python tools/make_public.py
"""
import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Kaldirilacak satirlar. Tam eslesme degil, icerik eslesmesi: satirda bu
# kaliplardan biri geciyorsa satir dusurulur.
PATTERNS = [
    re.compile(r"GIZLI\s*[-.]", re.I),
    re.compile(r"GİZLİ\s*[-.*]", re.I),
    re.compile(r"CONFIDENTIAL\s*[-—]", re.I),
    re.compile(r"patent basvurusuna kadar paylasmayin", re.I),
    re.compile(r"do not publish \(patent filing planned\)", re.I),
]

# README'deki uyari blogu ayrica ele alinir (cok satirli blok alintisi).
README_BLOCK = re.compile(
    r"\n> \*\*Status: private repository\.\*\*.*?\n(?=\n)", re.S
)


def hits(line: str) -> bool:
    return any(p.search(line) for p in PATTERNS)


def clean_text(path: pathlib.Path, dry: bool) -> int:
    src = path.read_text(encoding="utf-8")
    if path.name == "README.md":
        new = README_BLOCK.sub("\n", src, count=1)
    else:
        new = src
    kept, dropped = [], 0
    for line in new.splitlines(keepends=True):
        if hits(line):
            dropped += 1
            continue
        kept.append(line)
    new = "".join(kept)
    if new != src:
        print(f"  {path.relative_to(ROOT)}: {dropped} satir dusuruldu"
              + (" (+ README uyari blogu)" if path.name == "README.md"
                 and README_BLOCK.search(src) else ""))
        if not dry:
            path.write_text(new, encoding="utf-8")
        return 1
    return 0


def clean_notebook(path: pathlib.Path, dry: bool) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        kept = [ln for ln in src if not hits(ln)]
        if len(kept) != len(src):
            changed += len(src) - len(kept)
            cell["source"] = kept
    if changed:
        print(f"  {path.relative_to(ROOT)}: {changed} satir dusuruldu")
        if not dry:
            path.write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="dosyalari degistirme, sadece ne olacagini yaz")
    args = ap.parse_args()

    print("Gizlilik uyarilari taraniyor..." if args.dry_run
          else "Gizlilik uyarilari kaldiriliyor...")
    n = 0
    me = pathlib.Path(__file__).resolve()
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.resolve() == me:      # bu betik kendi uyarilarini silmez
            continue
        if path.suffix in {".py", ".md", ".txt", ".toml", ".cff", ".yml", ".yaml"}:
            n += clean_text(path, args.dry_run)
        elif path.suffix == ".ipynb":
            n += clean_notebook(path, args.dry_run)

    if n == 0:
        print("Temiz: kaldirilacak uyari kalmamis.")
    else:
        print(f"\n{n} dosya {'degisecek' if args.dry_run else 'degistirildi'}.")
        if args.dry_run:
            print("Uygulamak icin --dry-run olmadan calistirin.")
        else:
            print("Simdi README'nin ilk paragrafini bir kez gozden gecirin,")
            print("sonra depoyu herkese acik yapabilirsiniz.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
