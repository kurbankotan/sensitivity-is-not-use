"""Kisaltma sirasinda hicbir sayinin degismedigini dogrular.

Kural 14: makaledeki her sayi gercek olmali. Kisaltma metni degistirir ama
SAYILARI degistirmemelidir. Bu betik, kisaltma oncesi ve sonrasi arasinda
YENI bir sayinin ortaya cikmadigini garanti eder; kaybolan sayilar ise tek tek
listelenir ki her birinin kasitli oldugunu dogrulayabileyim.
"""
import docx, json, re, sys

DOC = "/mnt/user-data/outputs/EnsAff_Geometry_paper_EN.docx"
SNAP = "/home/claude/paper/.numbers.json"

def numbers(path):
    d = docx.Document(path)
    txt = "\n".join(p.text for p in d.paragraphs)
    for t in d.tables:
        for r in t.rows:
            for c in r.cells:
                txt += "\n" + c.text
    # kaynakca numaralari ve atif koseli parantezleri haric
    txt = re.sub(r"^\[\d+\] ", "", txt, flags=re.M)
    return sorted(set(re.findall(r"-?\d+\.?\d*", txt)))

if __name__ == "__main__":
    cur = numbers(DOC)
    if sys.argv[1:] == ["save"]:
        json.dump(cur, open(SNAP, "w"))
        print(f"anlik goruntu alindi: {len(cur)} farkli sayi")
    else:
        old = json.load(open(SNAP))
        new_ = [n for n in cur if n not in old]
        gone = [n for n in old if n not in cur]
        print(f"onceki {len(old)} -> simdiki {len(cur)} farkli sayi")
        print(f"YENI ORTAYA CIKAN (olmamali): {new_ if new_ else 'yok'}")
        print(f"kaybolan ({len(gone)}): {gone}")
        sys.exit(1 if new_ else 0)
