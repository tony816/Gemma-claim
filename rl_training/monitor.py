"""Compact read-only progress monitor; no calls that can wake a GPU worker."""
import json,time
from .core import OUT


def snapshot():
    result={}
    root=OUT/'pod'
    try:result['stage']=json.loads((root/'status.json').read_text(encoding='utf-8'))['stage']
    except (OSError,ValueError,KeyError):result['stage']='starting'
    path=root/'training.jsonl'
    if path.exists():
        lines=path.read_text(encoding='utf-8').splitlines()
        if lines:
            last=json.loads(lines[-1]);result.update(episodes=len(lines),step=last['step'],rewards=last['rewards'])
    result['evaluation_counts']={p.stem:len(p.read_text(encoding='utf-8').splitlines())
        for p in (root/'evaluation').glob('*.jsonl')}
    result['pod_absent']=(OUT/'termination.json').exists() or (OUT/'guard_finished.json').exists()
    return result


if __name__=='__main__':
    old=None;last=0
    while True:
        value=snapshot()
        if value!=old or time.time()-last>=60:
            print(json.dumps({'time':time.time(),**value}),flush=True);old=value;last=time.time()
        if value['pod_absent']:break
        time.sleep(5)
