"""Complete short-core original-audio evidence using the existing blind ASR workers."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import math
from src import late_audio as native
from src.contextual_review_contract import require

POLICY = {'version':'short-owner-core-1','target_seconds':6,'minimum_seconds':2,
          'maximum_seconds':8,'padding_seconds':.8,'maximum_asr_requests':800,
          'context':'','hotwords':[],'source_text_used':False,'frames_dropped':0}
RATE=native.ASR_RATE


def partition(owners, frames, speech):
    require(type(frames) is int and frames>0 and isinstance(owners,list) and owners,'Missing acoustic timeline')
    previous=0
    for segment in speech:
        require(set(segment)=={'start','end'} and all(type(v) is int for v in segment.values())
            and previous<=segment['start']<segment['end']<=frames,'Invalid VAD geometry')
        previous=segment['end']
    gaps=[(a['end']+b['start'])//2 for a,b in zip(speech,speech[1:]) if b['start']>a['end']]
    boundaries=[round(row['start']*RATE) for row in owners]+[frames]
    require(boundaries[0]==0,'Owner timeline does not begin at zero')
    for i,row in enumerate(owners):
        require(row['id']==i+1 and all(type(row[k]) in (int,float) and math.isfinite(row[k]) for k in ('start','end'))
            and row['start']<row['end'] and abs(round(row['end']*RATE)-boundaries[i+1])<=RATE*.0011,
            'Owners do not cover the continuous waveform')
    result=[]
    for owner,start,end in zip(owners,boundaries,boundaries[1:]):
        require(end-start>=2*RATE,'Owner too short for this declared policy')
        cuts=[start]
        while end-cuts[-1]>8*RATE:
            current=cuts[-1]; high=min(current+8*RATE,end-2*RATE)
            options=[x for x in gaps if current+2*RATE<=x<=high]
            cut=min(options,key=lambda x:(abs(x-current-6*RATE),x)) if options else min(current+6*RATE,high)
            cuts.append(cut)
        cuts.append(end)
        for number,(a,b) in enumerate(zip(cuts,cuts[1:]),1):
            require(2*RATE<=b-a<=8*RATE,'Invalid balanced short core')
            result.append({'view_id':f"short-o{owner['id']:04d}-c{number:03d}", 'owner_id':owner['id'],
                'owner_start':owner['start'],'owner_end':owner['end'],'kind':'raw_short',
                'core_start_frame':a,'core_end_frame':b,'crop_start_frame':max(0,a-round(.8*RATE)),
                'crop_end_frame':min(frames,b+round(.8*RATE)),'sample_rate':RATE,'duplicate_geometry':False})
    require(len(result)*2<=POLICY['maximum_asr_requests'],'Short acquisition exceeds declared call cap')
    require(result[0]['core_start_frame']==0 and result[-1]['core_end_frame']==frames
        and all(a['core_end_frame']==b['core_start_frame'] for a,b in zip(result,result[1:])),'Acoustic frames were dropped or duplicated')
    return result


def prepare(folder, original_preparation):
    """One CPU partition; no ASR models loaded. Reuse the preserved PCM master."""
    folder=Path(folder).resolve(); source=Path(original_preparation).resolve()
    if (folder/'preparation.json').exists(): return validate_preparation(folder)
    require(not folder.exists() or not any(folder.iterdir()),'Unregistered short acquisition exists')
    old=native._read(source);master=old['master'];mono=old['mono']
    for pin in (master['media'],master['master'],mono):native._check(pin)
    folder.mkdir(parents=True,exist_ok=True)
    native._write(folder/'preparation-started.json',{'original_preparation':native._pin(source),'policy':POLICY,'started_utc':native._now()})
    import soundfile as sf
    from faster_whisper.vad import VadOptions,get_speech_timestamps
    audio,rate=sf.read(mono['path'],dtype='float32')
    require(rate==RATE and len(audio)==mono['frames'] and audio.ndim==1,'Original mono geometry differs')
    speech=get_speech_timestamps(audio,VadOptions(**native.VAD_OPTIONS),sampling_rate=RATE)
    speech=[{k:int(row[k]) for k in ('start','end')} for row in speech]
    owners=old['request']['owners'];crops=partition(owners,mono['frames'],speech)
    request=deepcopy(old['request']);request.update(mode='blind',flagged_ids=None,masking_ids=None,bandit_manifest=None,worker_timeout=1800)
    preparation={'version':native.VERSION,'request':request,'master':master,'mono':mono,'crops':crops,
        'models':old['models'],'code_pins':[native._pin(native.__file__),native._pin(__file__),native._pin(native.PROJECT/'src/adjudicate.py'),native._pin(native.PROJECT/'src/local_backend.py')],
        'max_asr_requests':len(crops)*2,'runtime':native._runtime_identity(),'vad_options':native.VAD_OPTIONS,
        'vad_speech':speech,'short_policy':deepcopy(POLICY),'original_preparation':native._pin(source),'created_utc':native._now()}
    require(all(spec['available'] for spec in preparation['models'].values()),'Both declared ASR models must be installed')
    native._write(folder/'preparation.json',preparation);return validate_preparation(folder)


def validate_preparation(folder):
    folder=Path(folder).resolve();p=native._read(folder/'preparation.json')
    require(p['short_policy']==POLICY and p['request']['mode']=='blind' and p['request']['flagged_ids'] is None
        and p['request']['masking_ids'] is None and p['request']['bandit_manifest'] is None,'Short acoustic policy drift')
    old=native._read(native._check(p['original_preparation']))
    require(p['master']==old['master'] and p['mono']==old['mono'] and p['request']['owners']==old['request']['owners']
        and p['models']==old['models'],'Original acoustic identity changed')
    for pin in [p['master']['media'],p['master']['master'],p['mono'],*p['code_pins']]:native._check(pin)
    require(p['code_pins'][0]==native._pin(native.__file__) and native._pin(__file__) in p['code_pins'],'Short worker code unbound')
    expected=partition(p['request']['owners'],p['mono']['frames'],p['vad_speech'])
    require(expected==p['crops'] and p['max_asr_requests']==len(expected)*2,'Frozen partition changed')
    return p


def acquire(folder, *, timeout):
    folder=Path(folder).resolve();p=validate_preparation(folder)
    if (folder/'evidence.json').exists():return validate_evidence(folder)
    require(not (folder/'execution-started.json').exists(),'Started short acquisition cannot be repeated')
    require(timeout>1,'No audio execution budget')
    # The frozen worker timeout is bounded by the caller's stage deadline; the
    # surrounding alarm also terminates and joins an owned worker on expiry.
    require(native._runtime_identity()==p['runtime'],'ASR runtime changed')
    native._write(folder/'execution-started.json',{'preparation':native._pin(folder/'preparation.json'),'started_utc':native._now(),'remaining_seconds':timeout})
    native._execute(p,folder/'preparation.json',folder)
    return validate_evidence(folder)


def validate_evidence(folder):
    folder=Path(folder).resolve();p=validate_preparation(folder);result=native.validate_evidence(folder/'evidence.json')
    plan=native._read(native._check({'path':result['plan_path'],'sha256':result['plan_sha256']}))
    require(plan['crops']==p['crops'] and len(result['observations'])==p['max_asr_requests'],'Short evidence geometry or coverage changed')
    life=native._read(folder/'qwen-lifecycle.json')
    require(life.get('owned_workers_stopped') is True and life.get('restored') is True,'ASR backend not restored')
    return result
