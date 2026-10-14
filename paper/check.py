"""Makale uzerinde makine ile dogrulanabilir her kurali sinar."""
import docx, re, sys, unicodedata
D = "/mnt/user-data/outputs/EnsAff_Geometry_paper_EN.docx"
d = docx.Document(D)
paras = [p.text for p in d.paragraphs]
alltxt = "\n".join(paras)
for t in d.tables:
    for r in t.rows:
        for c in r.cells:
            alltxt += "\n" + c.text
FAIL = []
def ck(name, ok, extra=""):
    print(("  GECTI  " if ok else "  KALDI  ") + name + ("  " + extra if extra else ""))
    if not ok: FAIL.append(name)

ck("cozulmemis gonderme yok", not re.search(r"@[TF]?\{", alltxt),
   str(re.findall(r"@[TF]?\{[a-z_]+\}", alltxt)[:4]))
# Kural 13: uzun tire yok
bad = sorted({c for c in alltxt if c in "—–"})
ck("Kural 13: uzun/orta tire yok", not bad, repr(bad))
nonascii = sorted({c for c in alltxt if ord(c) > 127})
ck("yalnizca ASCII", not nonascii, repr(nonascii[:8]))
# numaralandirma
tn = [int(m) for m in re.findall(r"^Table (\d+)\.", "\n".join(paras), re.M)]
fn = [int(m) for m in re.findall(r"^Figure (\d+)\.", "\n".join(paras), re.M)]
ck("tablo basliklari 1..n sirali", tn == list(range(1, len(tn) + 1)), str(tn))
ck("sekil basliklari 1..n sirali", fn == list(range(1, len(fn) + 1)), str(fn))
refs = [int(m) for m in re.findall(r"^\[(\d+)\] ", "\n".join(paras), re.M)]
ck("kaynaklar 1..n bitisik", refs == list(range(1, len(refs) + 1)), f"{len(refs)} kaynak")
# metinde gecen her tablo/sekil numarasi var mi
# Basliklarin kendisi sayilmaz: bir basligin "Figure 3." ile baslamasi metinde
# ona atif yapildigi anlamina gelmez.
_body = "\n".join(p for p in paras if not re.match(r"^(Table|Figure) \d+\.", p))
used_t = {int(m) for m in re.findall(r"\bTable (\d+)\b", _body)}
used_f = {int(m) for m in re.findall(r"\bFigure (\d+)\b", _body)}
ck("her tabloya metinden atif var", set(tn) <= used_t, str(set(tn) - used_t))
ck("her sekle metinden atif var", set(fn) <= used_f, str(set(fn) - used_f))
ck("olmayan tabloya atif yok", used_t <= set(tn), str(used_t - set(tn)))
ck("olmayan sekle atif yok", used_f <= set(fn), str(used_f - set(fn)))
# metinde gecen her kaynak numarasi var
cited = {int(x) for m in re.findall(r"\[([\d, ]+)\]", alltxt) for x in m.split(",") if x.strip().isdigit()}
cited -= set(refs)          # kaynakcanin kendi numaralari
ck("olmayan kaynaga atif yok", not (cited - set(refs)), str(sorted(cited - set(refs))[:5]))
# bolum gondermeleri
secs = set(re.findall(r"^(\d+(?:\.\d+)?) ", "\n".join(paras), re.M))
refsec = set(re.findall(r"Section (\d+\.\d+)", alltxt))
ck("sarkan bolum gondermesi yok", refsec <= secs, str(sorted(refsec - secs)))
# gomulu gorsel sayisi
nimg = len(d.inline_shapes)
ck("gomulu gorsel = sekil sayisi", nimg == len(fn), f"{nimg} gorsel / {len(fn)} baslik")
words = len(re.findall(r"\b\w+\b", "\n".join(paras)))
print(f"\n  kelime (kaynakca dahil): {words}   tablo: {len(tn)}   sekil: {len(fn)}   kaynak: {len(refs)}")
print("\n" + ("TUM KONTROLLER GECTI" if not FAIL else f"KALAN: {FAIL}"))
sys.exit(1 if FAIL else 0)
