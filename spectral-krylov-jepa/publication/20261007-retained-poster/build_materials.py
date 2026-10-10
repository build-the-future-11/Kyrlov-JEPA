#!/usr/bin/env python3
"""Build a one-page presentation from pinned retained evidence; never train or evaluate models."""
import argparse,csv,hashlib,json,math,os,re,statistics
from pathlib import Path
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor,white
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from pypdf import PdfReader

FONT_DIR=Path(os.environ.get('CONFERENCE_FONT_DIR','/usr/share/fonts/truetype/dejavu'))
pdfmetrics.registerFont(TTFont('ConferenceSans',str(FONT_DIR/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('ConferenceSans-Bold',str(FONT_DIR/'DejaVuSans-Bold.ttf')))

NAVY=HexColor('#14283e'); TEAL=HexColor('#087f8c'); BLUE=HexColor('#456f9a'); GRAY=HexColor('#6d7883'); LIGHT=HexColor('#eef4f6')

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def text(c,s,x,y,w,size=18,color=NAVY,bold=False):
    style=ParagraphStyle('p',fontName='ConferenceSans-Bold' if bold else 'ConferenceSans',fontSize=size,leading=size*1.28,textColor=color,spaceAfter=0)
    p=Paragraph(s,style);_,h=p.wrap(w,2000);p.drawOn(c,x,y-h);return y-h

def block(c,title,body,x,y,w):
    y=text(c,title.upper(),x,y,w,18,TEAL,True)-12
    for para in body: y=text(c,para,x,y,w,17)-13
    return y

def load_claims(root,spec,manifest):
    for item in manifest['sources']:
        path=root/item['path']
        if digest(path)!=item['sha256']: raise ValueError('Pinned source differs: '+item['path'])
    if spec['project']=='lam':
        report=(root/manifest['sources'][0]['path']).read_text()
        counts={}
        for label,key in [('LAM','full'),('Matched supervised','matched')]:
            matches=[line for line in report.splitlines() if line.startswith('| '+label+' |')]
            if len(matches)!=1: raise ValueError('Ambiguous retained count table')
            counts[key]=[int(x) for x in re.findall(r'\d+',matches[0].split('|')[2])]
            if len(counts[key])!=5: raise ValueError('Five fixed seed counts required')
        audit=json.loads((root/manifest['sources'][1]['path']).read_text())
        n=audit['verifier']['validation_used_rows']
        if n!=295 or audit['verifier']['locked_test_evaluated']: raise ValueError('Frozen protocol boundary changed')
        rows=[{'label':label,'mean':statistics.mean(x/n for x in counts[key]),'sd':statistics.stdev(x/n for x in counts[key])} for label,key in [('Full LAM JEPA','full'),('Matched supervised','matched')]]
        if abs(rows[0]['mean']-audit['metrics']['full']['mean_accuracy'])>1e-12: raise ValueError('Retained full mean mismatch')
        difference=statistics.mean((a-b)/n for a,b in zip(counts['full'],counts['matched']))
        return {'rows':rows,'counts':counts,'n':n,'paired_difference':difference,'superiority_gate_passed':False,'planner':audit['metrics']['full_minus_no_planner'],'target':audit['metrics']['full_minus_no_target'],'scope':'retained report arithmetic; no raw prediction or checkpoint replay'}
    smoke=json.loads((root/manifest['sources'][0]['path']).read_text())
    audit=json.loads((root/manifest['sources'][1]['path']).read_text())
    rows=[]
    for key,label in [('krylov','Krylov +20 labels'),('scratch','Scratch +20 labels')]:
        m=smoke['steps']['eval_'+key+'_n20'];rows.append({'label':label,'fidelity':m['fidelity_mean'],'residual':m['residual_true_e_mean']})
    ids=None
    for method,subset,label in [('free_box','no_labels','Fixed free box'),('sine_ritz_9','no_labels','Nine mode Ritz'),('linear_ridge','n20','Ridge +20 labels')]:
        records=[r for r in audit['rows'] if r['method']==method and r['subset']==subset and r['split']=='test_ID']
        current={r['index'] for r in records}
        if len(records)!=12 or len(current)!=12 or (ids is not None and ids!=current): raise ValueError('Retained example identity mismatch')
        ids=current
        means={'fidelity':statistics.mean(r['fidelity'] for r in records),'residual':statistics.mean(r['residual_true_e'] for r in records)}
        summary=next(r for r in audit['summaries'] if r['method']==method and r['subset']==subset and r['split']=='test_ID')
        if abs(means['fidelity']-summary['fidelity_mean'])>1e-12 or abs(means['residual']-summary['residual_true_e_mean'])>1e-12: raise ValueError('Baseline summary mismatch')
        rows.append(dict(label=label,**means))
    contrast=next(r for r in audit['comparisons'] if r['neural_metrics']=='eval_krylov_n20/metrics.json' and r['baseline']=='free_box')
    return {'rows':rows,'test_indices':sorted(ids),'n':12,'paired_free_box_contrast':contrast,'scope':'neural aggregate readback plus classical per-example arithmetic; no model replay'}

def plot_lam(c,data,x,y,w,h):
    text(c,'Frozen validation accuracy',x,y,w,20,NAVY,True)
    top=y-45; bottom=top-h; bw=95; scale=h/.35
    for tick in [0,.1,.2,.3]:
        yy=bottom+tick*scale;c.setStrokeColor(HexColor('#dbe3e8'));c.line(x+45,yy,x+w-10,yy);text(c,f'{tick:.1f}',x,yy+9,38,13)
    for i,row in enumerate(data['rows']):
        xx=x+100+i*210; height=row['mean']*scale;c.setFillColor(TEAL if i==0 else BLUE);c.rect(xx,bottom,bw,height,stroke=0,fill=1)
        mid=xx+bw/2;low=bottom+(row['mean']-row['sd'])*scale;high=bottom+(row['mean']+row['sd'])*scale;c.setStrokeColor(NAVY);c.line(mid,low,mid,high);c.line(mid-10,high,mid+10,high);c.line(mid-10,low,mid+10,low)
        text(c,f"{row['mean']:.4f}",xx-15,high+27,140,19,NAVY,True);text(c,row['label'],xx-25,bottom-14,165,15)
    text(c,'Mean +/- sample SD across five fixed seeds. 295 validation items per seed.',x,bottom-58,w,14)
    return bottom-104

def plot_krylov(c,data,x,y,w,h):
    text(c,'High fidelity does not imply small physical error',x,y,w,20,NAVY,True)
    y-=44
    headers=['Method','Fidelity','Residual']
    cols=[x,x+270,x+395]
    for s,xx in zip(headers,cols):text(c,s,xx,y,150,15,GRAY,True)
    y-=28
    for i,row in enumerate(data['rows']):
        c.setFillColor(LIGHT if i%2==0 else white);c.rect(x-8,y-35,w,39,fill=1,stroke=0)
        text(c,row['label'],cols[0],y,265,17,TEAL if i==0 else NAVY,i==0)
        text(c,f"{row['fidelity']:.6f}",cols[1],y,120,17)
        text(c,f"{row['residual']:.6f}",cols[2],y,140,17)
        y-=43
    text(c,'Mean over twelve retained potentials. Residual: ||H psi - E_true psi|| / ||psi||, in energy units.',x,y-4,w,14)
    return y-72

def build(root,out):
    spec=json.loads((out/'poster_spec.json').read_text());manifest=json.loads((out/'source_manifest.json').read_text())
    abstract=(out/'abstract.txt').read_text().strip()
    if len(abstract.split())!=250: raise ValueError('Abstract must contain exactly250 whitespace-delimited words')
    data=load_claims(root,spec,manifest);(out/'claim_data.json').write_text(json.dumps(data,indent=2)+'\n')
    pdf=out/'poster.pdf';c=Canvas(str(pdf),pagesize=(1296,936),invariant=1,pageCompression=1)
    c.setTitle(spec['title']);c.setAuthor('Ryan Gomez');c.setSubject('Author review draft; retained evidence; nonarchival presentation candidate')
    c.setFillColor(NAVY);c.rect(0,762,1296,174,fill=1,stroke=0)
    text(c,spec['title'],42,904,1208,34,white,True)
    text(c,'Ryan Gomez  |  Author review draft  |  Retained evidence',44,794,1200,17,white)
    left=42;right=680;col=570;top=730
    y=block(c,'Research question',spec['question'],left,top,col)
    y=block(c,'Method and evidence boundary',spec['method'],left,y-15,col)
    y=block(c,'Why this matters for parsimony',spec['lesson'],left,y-15,col)
    y=block(c,'What remains unresolved',spec['limits'],left,y-8,col)
    z=(plot_lam if spec['project']=='lam' else plot_krylov)(c,data,right,top,col,190)
    z=block(c,'What the result supports',spec['result'],right,z-14,col)
    if min(y,z)<84:raise ValueError(f'Poster body exceeds safe footer: {y},{z}')
    c.setStrokeColor(HexColor('#cad6dc'));c.line(42,70,1254,70)
    text(c,spec['source_line'],42,57,1210,11)
    text(c,'AI assistance: Codex (GPT-6) supported source review, arithmetic checks, drafting and layout. Author review is required. No new outcome run.',42,34,1210,11)
    c.showPage();c.save()
    reader=PdfReader(str(pdf));extracted='\n'.join(p.extract_text() for p in reader.pages)
    if len(reader.pages)!=1 or 'AISTATS' in extracted or 'accepted' in extracted.lower(): raise ValueError('Unexpected presentation/branding state')
    checks={'stage':'PRESENTATION_DRAFT_PREPARED_NOT_SUBMITTED','abstract_words':250,'pages':1,'source_files_verified':len(manifest['sources']),'input_commit':manifest['input_commit'],'poster_sha256':digest(pdf),'claim_data_sha256':digest(out/'claim_data.json'),'minimum_body_y':min(y,z),'source_replayed':False,'new_outcome_runs':0,'author_review_complete':False,'whole_project_complete':False}
    (out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args();build(a.source_root,a.output)
