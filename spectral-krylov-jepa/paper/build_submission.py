"""Populate official template while preserving its styles and page geometry."""
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
from lxml import etree
from docx import Document
from docx.shared import Inches
ROOT=Path(__file__).resolve().parent
SUB=ROOT/'submission'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
W='{'+NS['w']+'}'
source=SUB/'CJSJ-template.docx'
with ZipFile(source) as z:
    parts={n:z.read(n) for n in z.namelist()}
xml=etree.fromstring(parts['word/document.xml'])
body=xml.find('w:body',NS)
paragraphs=body.findall('w:p',NS)
patterns={name:deepcopy(paragraphs[i]) for name,i in [('title',1),('author',2),('abstract',13),('heading',15),('body',17),('reference',31)]}
sect=deepcopy(body.find('w:sectPr',NS))
for child in list(body):body.remove(child)

def para(text,kind='body'):
    node=deepcopy(patterns[kind]); pp=node.find('w:pPr',NS)
    first=node.find('w:r',NS); rp=deepcopy(first.find('w:rPr',NS)) if first is not None and first.find('w:rPr',NS) is not None else None
    for c in list(node):
        if c is not pp:node.remove(c)
    r=etree.SubElement(node,W+'r')
    if rp is not None:r.append(rp)
    # All template normal body patterns retain the required TNR 10-point text.
    if kind=='body':
        rp=r.find('w:rPr',NS)
        if rp is None:rp=etree.SubElement(r,W+'rPr')
        for tag in ['sz','szCs']:
            e=rp.find(W+tag)
            if e is None:e=etree.SubElement(rp,W+tag)
            e.set(W+'val','20')
    t=etree.SubElement(r,W+'t');t.text=text;t.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
    body.append(node)

text=(SUB/'CJSJ_DRAFT.md').read_text()
reference=False; abstract=False
for block in text.strip().split('\n\n'):
    if block.startswith('# '):
        para(block[2:],'title');para('Author information pending human completion','author')
    elif block.startswith('## '):
        name=block[3:];reference=name=='References';abstract=name=='Abstract'
        if not abstract:para(name,'heading')
    else:
        if reference:
            import re
            block=re.sub(r'^\[\d+\]\s*', '', block)
        para(('Abstract — ' if abstract else '')+block.replace('\n',' '),'abstract' if abstract else ('reference' if reference else 'body'))
        abstract=False
body.append(sect)
parts['word/document.xml']=etree.tostring(xml,xml_declaration=True,encoding='UTF-8',standalone=True)
# Remove the template's invented sponsor and author footnote content explicitly.
if 'word/footnotes.xml' in parts:
    notes=etree.fromstring(parts['word/footnotes.xml'])
    for note in list(notes):
        if int(note.get(W+'id','0'))>0:notes.remove(note)
    parts['word/footnotes.xml']=etree.tostring(notes,xml_declaration=True,encoding='UTF-8',standalone=True)
# Clear author sample metadata rather than distributing template attribution.
if 'docProps/core.xml' in parts:
    core=etree.fromstring(parts['docProps/core.xml'])
    for e in core:
        if etree.QName(e).localname in ['creator','lastModifiedBy','title','subject','description']:
            e.text=''
    parts['docProps/core.xml']=etree.tostring(core,xml_declaration=True,encoding='UTF-8',standalone=True)
candidate=SUB/'Krylov_paper_draft.docx'
with ZipFile(candidate,'w',ZIP_DEFLATED) as z:
    for n,data in parts.items():z.writestr(n,data)
# Insert the real chart and caption as a cloned normal pattern; python-docx is
# used only for the drawing construction, then only modified document/media/rel
# parts are copied back to preserve opaque template parts.
d=Document(candidate)
p=d.add_paragraph();p.add_run().add_picture(str(ROOT.parent/'results/astra_baselines_20260927/baseline_infidelity.png'),width=Inches(3.35))
p=d.add_paragraph('Figure 1. Mean infidelity of simple baselines on the twelve retained smoke test-ID potentials. Lower is better. Linear and mean predictors use the specified nested training subsets. The classical sine controls use no labels. This is an exploratory baseline audit, not the unrun confirmatory experiment.')
for r in p.runs:
    from docx.shared import Pt
    r.font.name='Times New Roman';r.font.size=Pt(8)
tmp=SUB/'figure_insert.docx';d.save(tmp)
with ZipFile(tmp) as z:
    for n in z.namelist():
        if n in ['word/document.xml','word/_rels/document.xml.rels','[Content_Types].xml'] or n.startswith('word/media/'):
            parts[n]=z.read(n)
with ZipFile(candidate,'w',ZIP_DEFLATED) as z:
    for n,data in parts.items():z.writestr(n,data)
tmp.unlink()
with ZipFile(source) as z:
    unchanged=[n for n in z.namelist() if z.read(n)==parts[n]]
    changed=[n for n in z.namelist() if z.read(n)!=parts[n]]
(SUB/'template_fidelity.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'unchanged_parts':unchanged,'changed_parts':changed,'intentional_changes':['replace all example manuscript content','remove example footnotes and author metadata','add real figure and caption']},indent=2))
print(candidate)
