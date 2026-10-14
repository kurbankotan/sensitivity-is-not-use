"""Her sinifin __init__'i yalnizca kendi parametrelerini kullanmali.

Bir duzeltme yanlis sinifa dustugunde ortaya cikan hata sinifi burada yakalanir: v11'de B-faktor uyari
blogu SeqAffinityDataset'e dusmustu ve dizi modu calistirilsaydi NameError ile duserdi. Yalnizca yapi modu
kosuldugu icin dort surum boyunca gorunmedi.
"""
import ast, sys, glob, os

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ensaff")
bad = []
for f in sorted(glob.glob(os.path.join(SRC, "*.py"))):
    tree = ast.parse(open(f).read(), f)
    g = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    g |= {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    g |= {a.asname or a.name.split(".")[0] for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
          for a in n.names}
    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        for fn in [x for x in cls.body if isinstance(x, ast.FunctionDef)]:
            bound = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
            if fn.args.vararg: bound.add(fn.args.vararg.arg)
            if fn.args.kwarg: bound.add(fn.args.kwarg.arg)
            for node in ast.walk(fn):
                if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                    bound.add(node.id)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    bound |= {a.asname or a.name.split(".")[0] for a in node.names}
                elif isinstance(node, (ast.comprehension,)):
                    bound |= {t.id for t in ast.walk(node.target) if isinstance(t, ast.Name)}
                elif isinstance(node, ast.ExceptHandler) and node.name:
                    bound.add(node.name)
                elif isinstance(node, (ast.FunctionDef, ast.Lambda)) and node is not fn:
                    bound |= {a.arg for a in node.args.args + node.args.kwonlyargs}
                    if isinstance(node, ast.FunctionDef): bound.add(node.name)
            for node in ast.walk(fn):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) \
                        and node.id not in bound and node.id not in g and not hasattr(__builtins__, node.id):
                    bad.append(f"{os.path.basename(f)}:{node.lineno} {cls.name}.{fn.name} -> '{node.id}' tanimsiz")
for b in sorted(set(bad)):
    print("  HATA:", b)
print(f"\n{'BASARISIZ' if bad else 'GECTI'}: {len(set(bad))} tanimsiz ad")
sys.exit(1 if bad else 0)
