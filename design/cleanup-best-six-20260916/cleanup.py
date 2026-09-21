"""One-shot, manifest-bound project cleanup; never follow deletion symlinks."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, stat, time
ROOT=Path('/home/kciceblue/HF/auto-subtitle')
HERE=ROOT/'design/cleanup-best-six-20260916'
PLAN=HERE/'plan.json'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while chunk:=f.read(8*1024*1024):h.update(chunk)
 return h.hexdigest()
def inside(p):
 p=Path(os.path.abspath(p));assert p!=ROOT and p.is_relative_to(ROOT)
 assert not p.is_relative_to(ROOT/'.git') and not p.is_relative_to(ROOT/'.codex') and not p.is_relative_to(ROOT/'.agents')
 return p

def record(p,small_hash=False):
 s=p.lstat();v={'path':str(p),'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'mtime_ns':s.st_mtime_ns,'mode':s.st_mode,'links':s.st_nlink}
 if stat.S_ISLNK(s.st_mode):v['symlink']=os.readlink(p)
 elif small_hash and stat.S_ISREG(s.st_mode) and s.st_size<=2*1024*1024:v['sha256']=sha(p)
 return v

def leaves(p):
 if p.is_symlink() or not p.is_dir():yield p;return
 for folder,dirs,files in os.walk(p,followlinks=False):
  folder=Path(folder)
  for name in list(dirs):
   q=folder/name
   if q.is_symlink():yield q;dirs.remove(name)
  for name in files:yield folder/name

def build():
 assert not PLAN.exists()
 data=read(HERE/'keep-paths.json');code=read(HERE/'code-keep.json')
 whole=set(data['whole_directory_paths'])
 whole.update(str(ROOT/p) for p in ('input','ready_for_human_review','docs','models/anime-whisper','models/gemma4-31b-qat-q4','models/bandit-v2','.venv','.venv-voxtral','tests/fixtures','design/cleanup-best-six-20260916'))
 # Voxtral native assets are explicitly enumerated; its abandoned cache is removable.
 whole.add(str(ROOT/'models/voxtral-mini-4b-realtime-2602'))
 exact=set(data['all_direct_pins'])|set(data['exact_file_paths'])
 exact.update(code['keep_paths'])
 exact.update(str(ROOT/s.decode()) for s in Path('/tmp/auto-subtitle-cleanup-tracked.zlist').read_bytes().split(b'\0') if s)
 exact.update(str(ROOT/p) for p in ('profiles/selected-local.json','profiles/selected-writer.json','profiles/long-context-gemma.json','design/quality-trial-summary-20260916.md','design/quality-current-results-20260916.md','design/quality-loop-index-20260915.md'))
 # Preserve referents of retained symlinks as well as the link entries.
 link_records={};visited=set()
 while True:
  additions=[]
  for name in sorted(whole|exact):
   q=Path(name)
   if name in visited or not (q.exists() or q.is_symlink()):continue
   visited.add(name)
   for item in leaves(q):
    if not item.is_symlink():continue
    target=item.resolve(strict=False)
    link_records[str(item)]={'target':str(target),'exists':target.exists()}
    if target.exists():
     link_records[str(item)]['target_stat']=record(target,small_hash=target.is_file())
     if target.is_relative_to(ROOT) and target!=ROOT:additions.append((str(target),target.is_dir()))
  fresh=[(name,is_dir) for name,is_dir in additions if name not in whole|exact]
  if not fresh:break
  for name,is_dir in fresh:(whole if is_dir else exact).add(name)
 write(HERE/'retained-symlinks.json',link_records)
 protected={Path(s) for s in whole|exact if Path(s).is_relative_to(ROOT)}
 ancestors={a for p in protected for a in p.parents if a.is_relative_to(ROOT)}
 wholes={Path(s) for s in whole}
 def keep(p):return p in protected or any(p.is_relative_to(w) for w in wholes)
 scopes=[ROOT/s for s in ('archive','output','models/omnilingual-ctc-7b-v2','models/deepfilternet3','models/moss-transcribe-diarize','models/reading-hints','models/vibevoice-asr-hf','models/deepseek-v4-flash-0731','.venv-deepfilter','.venv-moss','.venv-omnilingual','.venv-reading','tmp','design','diagnostics','profiles','src','scripts','tests')]
 targets=[]
 def visit(p):
  if keep(p):return
  if p in ancestors:
   if p.is_dir() and not p.is_symlink():
    for q in sorted(p.iterdir()):visit(q)
   return
  targets.append(inside(p))
 for p in scopes:
  if p.exists() or p.is_symlink():visit(p)
 cache=ROOT/'models/voxtral-mini-4b-realtime-2602/.cache'
 if cache.exists():
  assert not any(p==cache or p.is_relative_to(cache) for p in exact_paths(exact))
  targets.append(cache)
 targets=sorted(set(targets))
 assert all(not a.is_relative_to(b) for a in targets for b in targets if a!=b)
 assert all(not (p==t or p.is_relative_to(t)) for p in protected for t in targets)
 listing=[]
 for t in targets:
  rows=[record(p) for p in leaves(t)];listing.append({'path':str(t),'kind':'symlink' if t.is_symlink() else 'directory' if t.is_dir() else 'file','root_stat':record(t),'files':rows,'logical_bytes':sum(r['bytes'] for r in rows),'allocated_bytes':sum(r['allocated_bytes'] for r in rows)})
 retained={}
 # Record the six results, required evidence/metadata and actual kept native assets.
 verify_whole=[Path(s) for s in data['whole_directory_paths']]+[ROOT/p for p in ('models/anime-whisper','models/gemma4-31b-qat-q4','models/voxtral-mini-4b-realtime-2602','models/bandit-v2','docs')]
 for p in [*(f for w in verify_whole for f in leaves(w)),*exact_paths(exact)]:
  if not p.is_relative_to(ROOT) or not (p.exists() or p.is_symlink()) or p.is_dir():continue
  if any(p==t or p.is_relative_to(t) for t in targets):continue
  retained[str(p)]=record(p,small_hash=True)
 write(HERE/'retained-before.json',retained)
 plan={'version':'best-six-cleanup-1','root':str(ROOT),'created_unix':time.time(),'keep_audit_sha256':sha(HERE/'keep-paths.json'),'code_audit_sha256':sha(HERE/'code-keep.json'),'retained_before_sha256':sha(HERE/'retained-before.json'),'retained_symlinks_sha256':sha(HERE/'retained-symlinks.json'),'targets':listing,'target_count':len(listing),'file_count':sum(len(x['files']) for x in listing),'allocated_bytes':sum(x['allocated_bytes'] for x in listing),'free_bytes_before':shutil.disk_usage(ROOT).free,'protected_whole_directories':sorted(whole),'protected_exact_files':sorted(exact)}
 write(PLAN,plan)
 print(json.dumps({k:plan[k] for k in ('target_count','file_count','allocated_bytes','free_bytes_before')}))

def exact_paths(items):return {Path(p) for p in items}
def comparable(rec):return {k:v for k,v in rec.items() if k!='links'}

def apply(expected):
 assert sha(PLAN)==expected and not (HERE/'completed.json').exists()
 plan=read(PLAN);assert plan['root']==str(ROOT) and plan['version']=='best-six-cleanup-1'
 assert sha(HERE/'keep-paths.json')==plan['keep_audit_sha256'] and sha(HERE/'code-keep.json')==plan['code_audit_sha256']
 assert sha(HERE/'retained-before.json')==plan['retained_before_sha256']
 before=read(HERE/'retained-before.json')
 assert sha(HERE/'retained-symlinks.json')==plan['retained_symlinks_sha256']
 links=read(HERE/'retained-symlinks.json')
 def verify_links():
  for name,entry in links.items():
   link=Path(name);assert link.is_symlink() and str(link.resolve(strict=False))==entry['target']
   target=Path(entry['target']);assert target.exists()==entry['exists'],'Retained symlink target disappeared'
   if target.exists() and target.is_file():assert comparable(record(target,'sha256' in entry['target_stat']))==comparable(entry['target_stat'])
 verify_links()
 for p,r in before.items():assert comparable(record(Path(p),'sha256' in r))==comparable(r),'Retained input changed before cleanup'
 for target in plan['targets']:
  p=inside(target['path']);assert comparable(record(p))==comparable(target['root_stat']),'Target changed before cleanup'
  actual={str(q):record(q) for q in leaves(p)};expected_files={r['path']:r for r in target['files']}
  assert actual==expected_files,'Deletion inventory changed'
  for protected in plan['protected_whole_directories']+plan['protected_exact_files']:
   q=Path(protected);assert not(q==p or q.is_relative_to(p)),'Protected path in deletion target'
 write(HERE/'started.json',{'plan_sha256':expected,'started_unix':time.time()})
 with (HERE/'deleted.jsonl').open('x') as log:
  for target in plan['targets']:
   p=inside(target['path'])
   if p.is_symlink() or not p.is_dir():p.unlink()
   else:shutil.rmtree(p)
   log.write(json.dumps({'path':str(p),'allocated_bytes':target['allocated_bytes']})+'\n');log.flush()
 failures=[]
 for p,r in before.items():
  try:
   if comparable(record(Path(p),'sha256' in r))!=comparable(r):failures.append(p)
  except OSError:failures.append(p)
 assert not failures,'Retained file changed after cleanup'
 verify_links()
 free=shutil.disk_usage(ROOT).free
 result={'plan_sha256':expected,'deleted_targets':len(plan['targets']),'deleted_files':plan['file_count'],'inventoried_allocated_bytes':plan['allocated_bytes'],'filesystem_free_bytes_before':plan['free_bytes_before'],'filesystem_free_bytes_after':free,'filesystem_free_bytes_delta':free-plan['free_bytes_before'],'retained_files_verified':len(before),'retained_small_hashes_verified':sum('sha256' in r for r in before.values()),'finished_unix':time.time()}
 write(HERE/'completed.json',result);print(json.dumps(result))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=('plan','apply'));parser.add_argument('--sha256');a=parser.parse_args()
 build() if a.action=='plan' else apply(a.sha256)
