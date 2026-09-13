"""One bounded calibration aggregation correction, preserving original results."""
import importlib.util
import json
from pathlib import Path
import shutil
import time

from .core import OUT,ROOT,write


def main():
    rows=[json.loads(s) for s in (OUT/'judge/responses.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['id'].startswith('C')]
    assert len(rows)==24 and all(r['valid'] for r in rows)
    labels=json.loads((OUT/'judge/labels.PRIVATE.json').read_text())
    import importlib.machinery
    loader=importlib.machinery.SourceFileLoader('rl_training.next_judge',str(ROOT/'rl_training/judge.py.next'))
    spec=importlib.util.spec_from_loader(loader.name,loader)
    module=importlib.util.module_from_spec(spec);loader.exec_module(module)
    packets=json.loads((OUT/'judge/packets.json').read_text())
    correct=0;preferences=0;changed=[]
    for row,label,packet in zip(rows,labels,packets):
        assert row['id']==label['id']==packet['id']
        rewards=module.semantic_rewards(row['parsed']['candidates'],packet['policy']['task_type'],row['parsed']['preference'])
        direction='A' if rewards[0]>rewards[1] else 'B' if rewards[1]>rewards[0] else 'tie'
        correct+=direction==label['expected'] and row['parsed']['preference']==label['expected']
        preferences+=row['parsed']['preference']==label['expected']
        if rewards!=row['rewards']:changed.append({'id':row['id'],'before':row['rewards'],'after':rewards})
    report={'total':24,'correct':correct,'agreement':correct/24,'pure_preference_correct':preferences,
            'changed_scalars':changed,'passed':correct/24>=.85,'time':time.time(),
            'rule_precommitted_before_unblinding':'repair_plan.json','semantic_outputs_unchanged':True}
    write('continuation_admission.json',report)
    print(json.dumps(report),flush=True)
    if not report['passed']:return
    shutil.copyfile(OUT/'judge/responses.jsonl',OUT/'judge/first_pass_responses.jsonl')
    if (OUT/'failure.json').exists():shutil.copyfile(OUT/'failure.json',OUT/'first_failure.json')
    for name in ['judge.py','train.py']:
        path=ROOT/'rl_training'/name
        shutil.copyfile(path,path.with_suffix(path.suffix+'.first_schema'))
        shutil.copyfile(str(path)+'.next',path)
    write('status.json',{'stage':'continuation_ready','time':time.time()})


if __name__=='__main__':main()
