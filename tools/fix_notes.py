"""Rewrite PowerPoint speaker notes as right-to-left Arabic paragraphs (one per line)."""
import sys, re, zipfile, shutil, os, html

src = sys.argv[1]
tmp = src + ".tmp"
zin = zipfile.ZipFile(src)
zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
pat = re.compile(r'(<p:sp>(?:(?!</p:sp>).)*?<p:ph type="body"(?:(?!</p:sp>).)*?<p:txBody>)(.*?)(</p:txBody>)', re.S)

def para(line):
    line = line.strip()
    head = line.startswith("سلايد") or line.startswith("الفكرة") or line.startswith("شو بحكي") \
        or line.startswith("لو سألوني") or line.startswith("الجواب")
    b = ' b="1"' if head else ""
    t = html.escape(line, quote=False)
    return (f'<a:p><a:pPr marR="0" algn="r" rtl="1"><a:spcAft><a:spcPts val="600"/></a:spcAft></a:pPr>'
            f'<a:r><a:rPr lang="ar-JO" altLang="en-US" sz="1600"{b} dirty="0"><a:latin typeface="Arial"/><a:cs typeface="Arial"/></a:rPr>'
            f'<a:t>{t}</a:t></a:r></a:p>')

for item in zin.infolist():
    data = zin.read(item.filename)
    if item.filename.startswith("ppt/notesSlides/notesSlide") and item.filename.endswith(".xml"):
        x = data.decode("utf8")
        def repl(m):
            texts = re.findall(r"<a:t>(.*?)</a:t>", m.group(2), re.S)
            raw = html.unescape("".join(texts))
            lines = [l for l in raw.split("\n") if l.strip()]
            if not lines:
                return m.group(0)
            body = '<a:bodyPr/><a:lstStyle/>' + "".join(para(l) for l in lines)
            return m.group(1) + body + m.group(3)
        x, n = pat.subn(repl, x, count=1)
        data = x.encode("utf8")
    zout.writestr(item, data)
zout.close(); zin.close()
shutil.move(tmp, src)
print("notes fixed:", src)
