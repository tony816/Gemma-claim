"""After rejected-checkpoint tests, export/reload the cumulative adapter, then finish.

No further training or changed selection rule. Avoid a large merged-weight upload
for a checkpoint that is explicitly ineligible for production deployment.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from .core import CONFIG, OUT, ROOT, RELEASE, sha, write


def state(stage, **extra):
    write('status.json',{'stage':stage,'time':time.time(),**extra})
    print(stage,flush=True)


def stop_finished_trainer():
    result = subprocess.run(['pgrep','-f','^/workspace/rl-venv/bin/python -u -m rl_training.train$'],
                            capture_output=True,text=True)
    ids = [int(x) for x in result.stdout.split()]
    assert len(ids)==1, 'expected_one_training_process'
    pid = ids[0]
    parent = int(Path(f'/proc/{pid}/stat').read_text().split()[3])
    parent_cmd = Path(f'/proc/{parent}/cmdline').read_bytes()
    assert b'/rl_training/boot.sh' in parent_cmd, 'unexpected_trainer_parent'
    # The boot trap must not overwrite recovery status after intentional takeover.
    os.kill(parent,signal.SIGSTOP)
    os.kill(pid,signal.SIGTERM)
    for _ in range(30):
        if not Path(f'/proc/{pid}').exists():break
        # Zombie means GPU is already released; the stopped parent cannot reap it.
        if Path(f'/proc/{pid}/stat').read_text().split()[2]=='Z':break
        time.sleep(1)
    else:os.kill(pid,signal.SIGKILL)
    os.kill(parent,signal.SIGKILL)


def run():
    plan = json.loads((OUT/'adapter_recovery_plan.json').read_text())
    while not plan.get('interrupt_after_calibration',False):
        current = json.loads((OUT/'status.json').read_text())
        if current['stage']=='merging':break
        if current['stage'] in {'failed','complete_training'}:return
        assert time.time()<plan['hard_deadline']-120, 'recovery_wait_deadline'
        time.sleep(1)
    selection=json.loads((OUT/'selection.json').read_text())
    assert selection['eligible_to_deploy'] is False
    if plan.get('interrupt_after_calibration',False):
        counts={}
        for label in ('rl','v2'):
            path=OUT/f'evaluation/{label}-test.jsonl'
            counts[label]=len(path.read_text().splitlines()) if path.exists() else 0
        write('test_interrupted_for_followup.json',{'counts':counts,
            'reason':'User funded a new trial after calibration rejection. Preserve this attempt, avoid completing a superseded final test. New training decisions use calibration only.',
            'test_results_used_for_selection':False,'final_test_complete':False})
    else:
        for label in ('rl','v2'):
            lines=(OUT/f'evaluation/{label}-test.jsonl').read_text().splitlines()
            assert len(lines)==20 and (OUT/f'evaluation/{label}-test-summary.json').exists()
    stop_finished_trainer()
    state('adapter_recovery')
    from huggingface_hub import HfApi,snapshot_download
    api=HfApi()
    checkpoint=json.loads((OUT/'latest_checkpoint.json').read_text())
    source=Path(checkpoint['path'])
    target=OUT/'adapter_export';target.mkdir(exist_ok=False)
    for name,expected in checkpoint['files'].items():
        assert sha(source/name)==expected
        if name not in {'resume.pt','README.md'}:shutil.copyfile(source/name,target/name)
    adapter_config=json.loads((target/'adapter_config.json').read_text())
    adapter_config['base_model_name_or_path']=CONFIG['base_model']
    adapter_config['revision']=CONFIG['base_revision']
    (target/'adapter_config.json').write_text(json.dumps(adapter_config,indent=2),encoding='utf-8')
    (target/'README.md').write_text('---\nlicense: gemma\nlibrary_name: peft\n'
        'base_model: google/gemma-4-31B-it\n---\n'
        '# Gemma claim RL v3 — rejected one-update pilot\n\n'
        '**Not approved for production deployment.** Calibration reward .785 -> .745; severe errors 5 -> 6.\n'
        'One on-policy group-normalized REINFORCE update from two sampled answers to one drawing episode, out of 88 available episodes.\n'
        'This is a cumulative v2 + RL adapter, not a separate RL-only delta.\n'
        f'Original v2: `{CONFIG["v2_model"]}` @ `{CONFIG["v2_revision"]}`.\n'
        f'Base: `{CONFIG["base_model"]}` @ `{CONFIG["base_revision"]}`. Historical v2 base-weight SHA could not be recovered.\n'
        'Use Gemma4ForConditionalGeneration with the pinned BF16 base, then PeftModel.from_pretrained with this adapter. '
        'The original v2 adapter is not additionally required.\n'
        'Frozen KR/US historical-case pilot only; automatic rubric scores do not establish legal accuracy. '
        'See evaluation evidence, original checkpoint optimizer/RNG state, and training source.\n',encoding='utf-8')
    hashes={p.name:sha(p) for p in target.iterdir() if p.is_file()}
    write('model_hashes.json',hashes)
    write('serving_format.json',{'format':'cumulative PEFT adapter including original v2 plus RL',
        'merged_export_cancelled':True,'reason':'Rejected calibration; preserve reloadable adapter within existing budget',
        'checkpoint_upload_seconds':136,'checkpoint_bytes_approx':1500000000,
        'eligible_to_deploy':False})
    api.upload_folder(repo_id=CONFIG['output_repo'],folder_path=ROOT/'rl_training',
        path_in_repo='training_source',ignore_patterns=['__pycache__/*'])
    api.upload_file(repo_id=CONFIG['output_repo'],path_or_fileobj=RELEASE/'manifest.json',
                    path_in_repo='training_materials/manifest.json')
    api.upload_folder(repo_id=CONFIG['output_repo'],folder_path=OUT,path_in_repo='evidence',
        ignore_patterns=['model/*','checkpoints/*','redownload/*','adapter_export/*','*.log','*.env'])
    commit=api.upload_folder(repo_id=CONFIG['output_repo'],folder_path=target,
        commit_message='Preserve rejected cumulative v2 plus one online RL update; adapter format')
    assert api.model_info(CONFIG['output_repo'],revision=commit.oid).private
    write('hub_upload.json',{'repo':CONFIG['output_repo'],'commit':commit.oid,'private':True,
        'format':'cumulative PEFT adapter','hashes':hashes})
    state('fresh_download_reload',commit=commit.oid)
    fresh=OUT/'redownload';assert not fresh.exists()
    snapshot_download(CONFIG['output_repo'],revision=commit.oid,local_dir=fresh,allow_patterns=list(hashes))
    assert all(sha(fresh/name)==expected for name,expected in hashes.items())
    import torch
    from transformers import AutoProcessor,Gemma4ForConditionalGeneration,set_seed
    from peft import PeftModel,get_peft_model_state_dict
    from safetensors.torch import load_file
    from .train import generate
    from .core import shuffled_train
    set_seed(CONFIG['seed'])
    processor=AutoProcessor.from_pretrained(fresh,local_files_only=True)
    base=Gemma4ForConditionalGeneration.from_pretrained(CONFIG['base_model'],revision=CONFIG['base_revision'],
        dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa')
    restored=PeftModel.from_pretrained(base,fresh,local_files_only=True,is_trainable=False)
    actual=get_peft_model_state_dict(restored)
    source_state=load_file(str(fresh/'adapter_model.safetensors'))
    assert set(actual)==set(source_state)
    assert all(torch.equal(actual[k].detach().cpu(),source_state[k].to(actual[k].dtype)) for k in actual)
    _,_,sample=generate(restored,processor,shuffled_train()[0])
    assert len(sample.strip())>20
    write('fresh_reload_verified.json',{'repo':CONFIG['output_repo'],'commit':commit.oid,
        'hashes_verified':len(hashes),'local_files_only_adapter':True,
        'base_model':CONFIG['base_model'],'base_revision':CONFIG['base_revision'],
        'actual_adapter_tensor_match':True,'raw_inference':sample,'eligible_to_deploy':False,
        'format':'cumulative PEFT adapter including v2 + RL, reloaded on pinned BF16 base'})
    api.upload_file(repo_id=CONFIG['output_repo'],path_or_fileobj=OUT/'fresh_reload_verified.json',
                    path_in_repo='evidence/fresh_reload_verified.json')
    state('complete_training',commit=commit.oid,eligible_to_deploy=False,steps=1,completed_train_episodes=1)


if __name__=='__main__':
    try:run()
    except BaseException as exc:
        write('recovery_failure.json',{'time':time.time(),'type':type(exc).__name__,'reason':str(exc)[:300]})
        state('failed',error_type=type(exc).__name__)
        raise SystemExit(1)
