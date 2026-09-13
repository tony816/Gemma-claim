"""RunPod-only online RL. No reference answer enters policy gradients."""
import gc
import json
import os
from pathlib import Path
import time
import traceback

import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoProcessor, Gemma4ForConditionalGeneration, set_seed

from .core import (CONFIG, ROOT, RELEASE, OUT, sha, write, append, rows,
                   verify_release, policy_inputs, advantages, completion_inputs,
                   rollout_loss, shuffled_train)
from .judge import Judge, calibrate


def status(stage, **extra):
    write('status.json', {'stage':stage,'time':time.time(),**extra})
    print(stage, json.dumps(extra), flush=True)


def state_copy(model):
    return {k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}


def generate(model, processor, policy, stochastic=False):
    encoded=policy_inputs(processor,policy).to(model.device)
    model.eval()
    kwargs={'do_sample':stochastic,'max_new_tokens':CONFIG['max_new_tokens'],'use_cache':True}
    if stochastic:kwargs.update(temperature=CONFIG['temperature'],top_p=CONFIG['top_p'])
    with torch.inference_mode():
        sequence=model.generate(**encoded,**kwargs)
    sequence=sequence.clone()  # ordinary tensors, eligible for later backward
    output=processor.decode(sequence[0,encoded['input_ids'].shape[-1]:],skip_special_tokens=True)
    return encoded,sequence,output


def evaluate(model, processor, judge, split, label):
    from serving.rl_client import resolve_response
    results=[]
    for policy in rows(f'policy/{split}.jsonl'):
        _,sequence,text=generate(model,processor,policy)
        judged=judge(policy,[text],f'{label}-{split}-{policy["episode_id"]}')
        try:
            resolved=resolve_response(text,policy,allow_code_fence=True,source_split=split)
            structure={'valid':True,'resolved':resolved}
        except Exception as exc:
            structure={'valid':False,'error_type':type(exc).__name__}
        item={'episode_id':policy['episode_id'],'task_type':policy['task_type'],
              'raw':text,'reward':judged['rewards'][0],
              'judgment':judged['parsed']['candidates'][0],'structure':structure}
        append(f'evaluation/{label}-{split}.jsonl',item);results.append(item)
        print('EVAL',label,split,len(results),flush=True)
        del sequence
    summary={}
    for task in ['all','claim_set','patent_advisory']:
        selected=[r for r in results if task=='all' or r['task_type']==task]
        summary[task]={'n':len(selected),'mean_reward':sum(r['reward'] for r in selected)/len(selected),
            'severe_errors':sum(r['judgment']['severe_error'] for r in selected),
            'resolved_structure':sum(r['structure']['valid'] for r in selected)}
    write(f'evaluation/{label}-{split}-summary.json',summary)
    return summary


def ensure_repo():
    api=HfApi()
    from huggingface_hub.errors import RepositoryNotFoundError
    try:
        info=api.model_info(CONFIG['output_repo'])
    except RepositoryNotFoundError:
        api.create_repo(CONFIG['output_repo'],private=True,exist_ok=False)
    else:
        marker=OUT/'repo_owned.json'
        assert marker.exists() and info.private, 'refuse_unrelated_existing_repository'
    write('repo_owned.json',{'repo':CONFIG['output_repo'],'private':True,'run_id':CONFIG['run_id']})
    assert api.model_info(CONFIG['output_repo']).private
    return api


def checkpoint(model, processor, optimizer, episode, step):
    target=OUT/f'checkpoints/step-{step:04d}'
    target.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(target,safe_serialization=True,selected_adapters=['default'])
    processor.save_pretrained(target)
    torch.save({'optimizer':optimizer.state_dict(),'episode':episode,'step':step,
        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},target/'resume.pt')
    files={p.name:sha(p) for p in target.iterdir() if p.is_file()}
    write(f'checkpoints/step-{step:04d}/hashes.json',files)
    # Reload real saved tensors on CPU before allowing a safety backup to count.
    from safetensors.torch import load_file
    restored=load_file(str(target/'adapter_model.safetensors'))
    actual=state_copy(model)
    assert set(actual)==set(restored)
    assert all(torch.equal(actual[k],restored[k]) for k in actual)
    api=ensure_repo()
    commit=api.upload_folder(repo_id=CONFIG['output_repo'],folder_path=target,
        path_in_repo=f'{CONFIG.get("checkpoint_prefix", "checkpoints")}/step-{step:04d}',commit_message=f'Online RL cumulative v2 checkpoint step {step}')
    write('latest_checkpoint.json',{'step':step,'episode':episode,'path':str(target),
        'commit':commit.oid,'files':files,'cumulative_v2':True,'cpu_tensor_reload':True})
    return target


def run():
    set_seed(CONFIG['seed'])
    OUT.mkdir(parents=True,exist_ok=True)
    write('run_config.json',CONFIG);write('release_verified.json',verify_release())
    write('runtime.json',{'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0),'vram':torch.cuda.get_device_properties(0).total_memory})
    status('judge_loading')
    judge=Judge()
    status('judge_calibration')
    calibrate(judge)
    status('v2_loading')
    processor=AutoProcessor.from_pretrained(CONFIG['base_model'],revision=CONFIG['base_revision'])
    adapter=hf_hub_download(CONFIG['v2_model'],'adapter_model.safetensors',revision=CONFIG['v2_revision'])
    assert sha(adapter)==CONFIG['v2_weight_sha256']
    base=Gemma4ForConditionalGeneration.from_pretrained(CONFIG['base_model'],revision=CONFIG['base_revision'],
        dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa')
    model=PeftModel.from_pretrained(base,CONFIG['v2_model'],revision=CONFIG['v2_revision'],is_trainable=True)
    trainable=[p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable)==122429440
    original=state_copy(model)
    from safetensors.torch import load_file
    source_state=load_file(adapter)
    assert set(original)==set(source_state)
    assert all(torch.equal(original[k],source_state[k].to(original[k].dtype)) for k in original)
    del source_state
    write('v2_loaded.json',{'model':CONFIG['v2_model'],'revision':CONFIG['v2_revision'],
        'adapter_sha256':sha(adapter),'trainable_parameters':sum(p.numel() for p in trainable),
        'base_revision':CONFIG['base_revision'],'historical_base_revision_recovered':False})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    status('baseline_calibration')
    baseline_start=time.time()
    baseline=evaluate(model,processor,judge,'calibration','v2')
    baseline_seconds=time.time()-baseline_start
    # Reserve measured time for three equally sized held-out evaluation passes,
    # HF merge/upload/reload, and the final checkpoint copy, before more RL.
    final_reserve=3.2*baseline_seconds+780
    write('time_reserve.json',{'baseline_seconds':baseline_seconds,'final_reserve_seconds':final_reserve})
    optimizer=torch.optim.AdamW(trainable,lr=CONFIG['learning_rate'],weight_decay=0.)
    status('online_rl')
    deadline=float(os.environ['RL_STARTED_AT'])+CONFIG['training_soft_deadline_seconds']
    step=0;zeros=0;latest=None;completed_episodes=0
    for episode,policy in enumerate(shuffled_train(),1):
        if time.time()>deadline:break
        if time.time()+final_reserve>float(os.environ['RL_STARTED_AT'])+CONFIG['hard_pod_deadline_seconds']:break
        start=time.time();sequences=[];texts=[];encoded=None
        for _ in range(CONFIG['group_size']):
            encoded,sequence,text=generate(model,processor,policy,stochastic=True)
            sequences.append(sequence);texts.append(text)
        judged=judge(policy,texts,f'train-{episode:03d}')
        rewards=judged['rewards'];advs=advantages(rewards)
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        if advs.abs().max()>0:
            model.train()
            for sequence,adv in zip(sequences,advs):
                inputs=completion_inputs(encoded,sequence)
                n=sequence.shape[-1]-encoded['input_ids'].shape[-1]
                outputs=model(**inputs,use_cache=False,logits_to_keep=n+1)
                loss=rollout_loss(outputs.logits,sequence[:,-n:],adv)/CONFIG['group_size']
                assert torch.isfinite(loss), 'nonfinite_loss'
                loss.backward();losses.append(float(loss.detach()))
                del outputs,loss,inputs
            norm=torch.nn.utils.clip_grad_norm_(trainable,CONFIG['gradient_clip'])
            assert torch.isfinite(norm) and norm>0, 'invalid_gradient'
            optimizer.step();step+=1;zeros=0
            changed=sum(not torch.equal(original[k],v) for k,v in state_copy(model).items())
            assert changed>0,'no_v2_weight_change'
        else:
            norm=torch.tensor(0.);changed=None;zeros+=1
        record={'episode':episode,'step':step,'episode_id':policy['episode_id'],'raw':texts,
            'rewards':rewards,'advantages':advs.tolist(),'loss':sum(losses),
            'gradient_norm':float(norm),'changed_v2_tensors':changed,'seconds':time.time()-start,
            'gpu_peak_bytes':torch.cuda.max_memory_allocated()}
        append('training.jsonl',record);print('TRAIN',episode,step,rewards,flush=True)
        completed_episodes=episode
        if step and (latest is None or step%CONFIG['checkpoint_every']==0):
            if latest is None or latest.name!=f'step-{step:04d}':
                latest=checkpoint(model,processor,optimizer,episode,step)
        del encoded,sequences
        if zeros>=CONFIG['zero_advantage_stop']:break
    assert step>0,'no_online_rl_updates'
    if latest is None or latest.name!=f'step-{step:04d}':
        latest=checkpoint(model,processor,optimizer,completed_episodes,step)
    status('checkpoint_selection',steps=step,episodes=completed_episodes)
    trained=evaluate(model,processor,judge,'calibration','rl')
    eligible=(trained['all']['mean_reward']>=baseline['all']['mean_reward'] and
              trained['all']['severe_errors']<=baseline['all']['severe_errors'])
    write('selection.json',{'selected_checkpoint':latest.name,'eligible_to_deploy':eligible,
        'rule':'Final RL checkpoint, require nondecreasing mean semantic reward and no extra severe calibration errors',
        'baseline':baseline,'rl':trained,'test_access_before_selection':False})
    trained_state=state_copy(model)
    status('final_test')
    evaluate(model,processor,judge,'test','rl')
    set_peft_model_state_dict(model,original)
    evaluate(model,processor,judge,'test','v2')
    set_peft_model_state_dict(model,trained_state)
    del original,trained_state,optimizer,judge
    gc.collect();torch.cuda.empty_cache()
    status('merging')
    merged=model.merge_and_unload(safe_merge=True)
    export=OUT/'model';export.mkdir(exist_ok=True)
    merged.save_pretrained(export,safe_serialization=True,max_shard_size='5GB')
    processor.save_pretrained(export)
    # Preserve original architectural config spelling for older vLLM loaders;
    # Transformers 5 also reads this exact pinned original configuration.
    import shutil
    shutil.copyfile(hf_hub_download(CONFIG['base_model'],'config.json',revision=CONFIG['base_revision']),
                    export/'config.json')
    (export/'README.md').write_text('---\nlicense: gemma\nbase_model: google/gemma-4-31B-it\n---\n'
        '# Gemma claim RL v3 pilot\n\nMerged v2 plus online group-normalized REINFORCE updates.\n'
        'Small KR/US historical-case pilot; not a guarantee of legal advice quality.\n'
        f'Origin v2: {CONFIG["v2_revision"]}. Base: {CONFIG["base_revision"]}.\n'
        f'Deployment calibration gate: {eligible}. See evidence and cumulative checkpoints.\n',encoding='utf-8')
    hashes={p.name:sha(p) for p in export.iterdir() if p.is_file()}
    write('model_hashes.json',hashes)
    api=ensure_repo()
    api.upload_folder(repo_id=CONFIG['output_repo'],folder_path=ROOT/'rl_training',
                      path_in_repo='training_source',ignore_patterns=['__pycache__/*'])
    api.upload_folder(repo_id=CONFIG['output_repo'],folder_path=OUT,path_in_repo='evidence',
        ignore_patterns=['model/*','checkpoints/*','redownload/*','*.log','*.env'])
    commit=api.upload_folder(repo_id=CONFIG['output_repo'],folder_path=export,
                            commit_message=f'Merged cumulative v2 plus online RL {step} updates')
    write('hub_upload.json',{'repo':CONFIG['output_repo'],'commit':commit.oid,'private':True,'hashes':hashes})
    del merged,model,base,trainable
    gc.collect();torch.cuda.empty_cache()
    status('fresh_download_reload',commit=commit.oid)
    fresh=OUT/'redownload'
    assert not fresh.exists()
    snapshot_download(CONFIG['output_repo'],revision=commit.oid,local_dir=fresh,allow_patterns=list(hashes))
    assert all(sha(fresh/name)==expected for name,expected in hashes.items())
    reprocessor=AutoProcessor.from_pretrained(fresh,local_files_only=True)
    restored=Gemma4ForConditionalGeneration.from_pretrained(fresh,local_files_only=True,
        dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa')
    _,_,sample=generate(restored,reprocessor,shuffled_train()[0])
    assert len(sample.strip())>20
    write('fresh_reload_verified.json',{'repo':CONFIG['output_repo'],'commit':commit.oid,
        'hashes_verified':len(hashes),'local_files_only':True,'raw_inference':sample,
        'eligible_to_deploy':eligible,'format':'full bf16 merged cumulative v2 plus RL'})
    api.upload_file(repo_id=CONFIG['output_repo'],path_or_fileobj=OUT/'fresh_reload_verified.json',
                    path_in_repo='evidence/fresh_reload_verified.json')
    status('complete_training',commit=commit.oid,eligible_to_deploy=eligible,steps=step)


if __name__=='__main__':
    try:run()
    except BaseException as exc:
        # No secrets in formatted error output; backend exceptions may contain URLs.
        write('failure.json',{'type':type(exc).__name__,'reason':str(exc)[:500],
            'traceback_frames':traceback.format_tb(exc.__traceback__),'time':time.time()})
        status('failed',error_type=type(exc).__name__)
        raise SystemExit(1)
