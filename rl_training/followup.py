"""Bounded second pilot, original v2 policy and frozen-v2 KL reference."""
import gc,json,os,random,shutil,time,traceback
from pathlib import Path
import torch
from huggingface_hub import HfApi,hf_hub_download,snapshot_download
from peft import PeftModel,get_peft_model_state_dict,set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoProcessor,Gemma4ForConditionalGeneration,set_seed
from .core import CONFIG,OUT,ROOT,RELEASE,write,append,sha,digest,rows,verify_release,completion_inputs
from .train import status,state_copy,generate,evaluate,checkpoint,ensure_repo
from .judge import Judge,SYSTEM
from .followup_math import rewards_for,weighted_advantages,token_logps,objective,activate


def balanced_train():
    from serving.rl_client import evidence_in_policy
    buckets={k:[] for k in ['drawing','annotation','KR','US']}
    for policy in rows('policy/train.jsonl'):
        if policy['task_type']=='claim_set':key='annotation' if evidence_in_policy(policy)[0] else 'drawing'
        else:key='KR' if policy['episode_id'].startswith('KR-') else 'US'
        buckets[key].append(policy)
    rng=random.Random(CONFIG['seed'])
    for bucket in buckets.values():rng.shuffle(bucket)
    result=[]
    while any(buckets.values()):
        for bucket in buckets.values():
            if bucket:result.append(bucket.pop())
    assert len(result)==88 and len({p['episode_id'] for p in result})==88
    return result[:CONFIG['max_episodes']]


def eligible(summary, baseline):
    return (summary['all']['mean_reward']>=baseline['all']['mean_reward'] and
        summary['all']['severe_errors']<=baseline['all']['severe_errors'] and
        summary['all']['resolved_structure']>=baseline['all']['resolved_structure'])


def export_reload(model,processor,latest,selected,steps):
    target=OUT/'adapter_export';target.mkdir(exist_ok=False)
    model.save_pretrained(target,safe_serialization=True,selected_adapters=['default'])
    processor.save_pretrained(target)
    config=json.loads((target/'adapter_config.json').read_text())
    config.update(base_model_name_or_path=CONFIG['base_model'],revision=CONFIG['base_revision'])
    (target/'adapter_config.json').write_text(json.dumps(config,indent=2))
    (target/'README.md').write_text('---\nlicense: gemma\nlibrary_name: peft\nbase_model: google/gemma-4-31B-it\n---\n'
        '# Gemma claim online RL v3 pilot\n\n'
        f'Calibration deployment gate: **{selected["eligible_to_deploy"]}**. Selected update: {steps}.\n'
        'Cumulative v2 plus online group-relative policy-gradient updates with a frozen v2 KL reference. '
        'Original v2 is included in this adapter and is not separately required at inference.\n'
        f'Origin `{CONFIG["v2_model"]}` @ `{CONFIG["v2_revision"]}`.\n'
        f'Pinned BF16 base `{CONFIG["base_model"]}` @ `{CONFIG["base_revision"]}`. '
        'Historical v2 base-weight revision was not recoverable; this compatible revision was verified.\n'
        'Frozen small KR/US historical-case pilot; no claim of general legal accuracy. '
        f'See runs/{CONFIG["run_id"]} for the manifest, selection, calibration/test outputs, '
        'resume checkpoints, source, and independent judge limitations.\n',encoding='utf-8')
    hashes={p.name:sha(p) for p in target.iterdir() if p.is_file()}
    write('model_hashes.json',hashes)
    api=ensure_repo();prefix='runs/'+CONFIG['run_id']
    api.upload_folder(repo_id=CONFIG['output_repo'],folder_path=ROOT/'rl_training',path_in_repo=prefix+'/training_source',ignore_patterns=['__pycache__/*'])
    api.upload_file(repo_id=CONFIG['output_repo'],path_or_fileobj=RELEASE/'manifest.json',path_in_repo=prefix+'/manifest.json')
    api.upload_folder(repo_id=CONFIG['output_repo'],folder_path=OUT,path_in_repo=prefix+'/evidence',
        ignore_patterns=['checkpoints/*','resume_checkpoint/*','adapter_export/*','redownload/*','*.log','*.env'])
    commit=api.upload_folder(repo_id=CONFIG['output_repo'],folder_path=target,commit_message=f'Cumulative v2 plus selected online RL step {steps}')
    assert api.model_info(CONFIG['output_repo'],revision=commit.oid).private
    write('hub_upload.json',{'repo':CONFIG['output_repo'],'commit':commit.oid,'private':True,'hashes':hashes})
    # Release all original modules before actual full-model reconstruction.
    return commit.oid,hashes


def run():
    set_seed(CONFIG['seed']);write('run_config.json',CONFIG);write('release_verified.json',verify_release())
    precommit=json.loads((OUT/'precommit.json').read_text())
    assert precommit['test_used_for_selection'] is False
    write('runtime.json',{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda})
    status('judge_loading');judge=Judge()
    old=json.loads((OUT/'prior/judge/configuration.json').read_text())
    current=json.loads((OUT/'judge/configuration.json').read_text())
    assert old==current,'judge_changed_requires_new_blind_calibration'
    write('judge/calibration_reuse.json',{'source_run':'rl_v3_runpod_evidence','actual_RunPod_comparisons':24,
        'agreement':1.,'unchanged_configuration':True,'system_sha256':digest(SYSTEM),
        'cpu_contract_wrapper_agreement':24,'note':'Same actual GPU judge, not prior materials audit; reused without new inference'})
    status('v2_loading')
    processor=AutoProcessor.from_pretrained(CONFIG['base_model'],revision=CONFIG['base_revision'])
    adapter=hf_hub_download(CONFIG['v2_model'],'adapter_model.safetensors',revision=CONFIG['v2_revision'])
    assert sha(adapter)==CONFIG['v2_weight_sha256']
    base=Gemma4ForConditionalGeneration.from_pretrained(CONFIG['base_model'],revision=CONFIG['base_revision'],
        dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa')
    model=PeftModel.from_pretrained(base,CONFIG['v2_model'],revision=CONFIG['v2_revision'],is_trainable=True)
    model.load_adapter(CONFIG['v2_model'],adapter_name='v2_reference',revision=CONFIG['v2_revision'],is_trainable=False)
    activate(model,'default');original=state_copy(model);source=load_file(adapter)
    assert set(original)==set(source) and all(torch.equal(original[k],source[k].to(original[k].dtype)) for k in original)
    refstate=get_peft_model_state_dict(model,adapter_name='v2_reference')
    assert all(torch.equal(original[k],refstate[k].detach().cpu()) for k in original)
    del source,refstate
    trainable=[p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable)==122429440
    write('v2_loaded.json',{'revision':CONFIG['v2_revision'],'adapter_sha256':sha(adapter),
        'base_revision':CONFIG['base_revision'],'exact_initial_tensor_match':True,'frozen_reference_exact_v2':True,
        'trainable_parameters':sum(p.numel() for p in trainable)})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    baseline=json.loads((OUT/'prior/evaluation/v2-calibration-summary.json').read_text())
    (OUT/'evaluation').mkdir(exist_ok=True)
    for name in ['v2-calibration.jsonl','v2-calibration-summary.json']:
        shutil.copyfile(OUT/'prior/evaluation'/name,OUT/'evaluation'/name)
    write('baseline_reuse.json',{'original_v2_same_weights':True,'same_generation_max_new_tokens':768,
        'do_sample':False,'same_policy_inputs':True,'same_judge':True,'source_run':'rl_v3_runpod_evidence'})
    optimizer=torch.optim.AdamW(trainable,lr=CONFIG['learning_rate'],weight_decay=0.)
    start=float(os.environ['RL_STARTED_AT']);step=0;latest=None;selected=None;selected_state=None;zeros=0
    resume_episode=0
    if CONFIG.get('resume_from'):
        status('resume_loading')
        from .resume_state import from_hub
        restored_state=from_hub(model,optimizer)
        step=restored_state['step'];resume_episode=restored_state['episode']
        selected=json.loads((OUT/'prior/step8_selection.json').read_text())
        assert selected['step']==step and selected['eligible_to_deploy'] is False
        selected_state=state_copy(model)
        for suffix in ['.jsonl','-summary.json']:
            shutil.copyfile(OUT/f'prior/evaluation/step-8-calibration{suffix}',OUT/f'evaluation/step-8-calibration{suffix}')
        append('selection_candidates.jsonl',selected)
    status('online_rl')
    for episode,policy in enumerate(balanced_train(),1):
        if episode<=resume_episode:continue
        if time.time()>start+CONFIG['training_soft_deadline_seconds'] or step>=CONFIG['max_updates']:break
        tick=time.time();sequences=[];texts=[]
        for _ in range(CONFIG['group_size']):
            encoded,sequence,text=generate(model,processor,policy,True);sequences.append(sequence);texts.append(text)
        assessed=judge(policy,texts,f'train-{episode:03d}')
        rewards,contract_errors=rewards_for(policy,texts,assessed['parsed']);advs=weighted_advantages(rewards)
        optimizer.zero_grad(set_to_none=True);losses=[];kls=[];norm=torch.tensor(0.)
        if advs.abs().max()>0:
            for sequence,adv in zip(sequences,advs):
                inputs=completion_inputs(encoded,sequence);n=sequence.shape[1]-encoded['input_ids'].shape[1]
                activate(model,'v2_reference');model.eval()
                with torch.no_grad():
                    ref=token_logps(model(**inputs,use_cache=False,logits_to_keep=n+1).logits,sequence[:,-n:])
                activate(model,'default');model.train()
                outputs=model(**inputs,use_cache=False,logits_to_keep=n+1)
                loss,kl=objective(outputs.logits,sequence[:,-n:],adv,ref,CONFIG['kl_beta'])
                loss=loss/CONFIG['group_size'];assert torch.isfinite(loss)
                loss.backward();losses.append(float(loss.detach()));kls.append(float(kl.detach()))
                del inputs,outputs,loss,kl,ref
            norm=torch.nn.utils.clip_grad_norm_(trainable,CONFIG['gradient_clip']);assert torch.isfinite(norm) and norm>0
            optimizer.step();step+=1;zeros=0
        else:zeros+=1
        current=state_copy(model);changed=sum(not torch.equal(original[k],v) for k,v in current.items());del current
        if step:assert changed>0
        append('training.jsonl',{'episode':episode,'step':step,'episode_id':policy['episode_id'],
            'raw':texts,'rewards':rewards,'semantic_rewards':assessed['rewards'],'contract_errors':contract_errors,
            'advantages':advs.tolist(),'loss':sum(losses),'sampled_kl':kls,'gradient_norm':float(norm),
            'changed_v2_tensors':changed,'seconds':time.time()-tick,'gpu_peak_bytes':torch.cuda.max_memory_allocated()})
        print('TRAIN',episode,step,rewards,flush=True)
        if step and (latest is None or step in CONFIG['selection_steps']):
            if latest is None or latest.name!=f'step-{step:04d}':latest=checkpoint(model,processor,optimizer,episode,step)
        del encoded,sequences,sequence
        if step in CONFIG['selection_steps'] and not (OUT/f'evaluation/step-{step}-calibration-summary.json').exists():
            status('checkpoint_selection',step=step)
            summary=evaluate(model,processor,judge,'calibration',f'step-{step}')
            candidate={'step':step,'checkpoint':latest.name,'eligible_to_deploy':eligible(summary,baseline),'baseline':baseline,'rl':summary,
                'rule':precommit['selection_rule'],'test_access_before_selection':False}
            append('selection_candidates.jsonl',candidate)
            if selected is None or summary['all']['mean_reward']>selected['rl']['all']['mean_reward']:
                selected=candidate;selected_state=state_copy(model)
            if candidate['eligible_to_deploy']:
                selected=candidate;selected_state=state_copy(model);break
            status('online_rl')
        if zeros>=CONFIG['zero_advantage_stop']:break
    assert step>0,'no_online_rl_updates'
    if latest is None or latest.name!=f'step-{step:04d}':latest=checkpoint(model,processor,optimizer,episode,step)
    if selected is None:
        status('checkpoint_selection',step=step)
        summary=evaluate(model,processor,judge,'calibration',f'step-{step}')
        selected={'step':step,'checkpoint':latest.name,'eligible_to_deploy':eligible(summary,baseline),'baseline':baseline,'rl':summary,
            'rule':precommit['selection_rule'],'test_access_before_selection':False}
        selected_state=state_copy(model)
    write('selection.json',selected)
    set_peft_model_state_dict(model,selected_state)
    for suffix in ['.jsonl','-summary.json']:
        shutil.copyfile(OUT/f'evaluation/step-{selected["step"]}-calibration{suffix}',OUT/f'evaluation/rl-calibration{suffix}')
    refstate=get_peft_model_state_dict(model,adapter_name='v2_reference')
    assert all(torch.equal(original[k],refstate[k].detach().cpu()) for k in original)
    write('reference_unchanged.json',{'exact_tensors':len(original),'passed':True})
    del refstate
    status('final_test');evaluate(model,processor,judge,'test','rl')
    set_peft_model_state_dict(model,original);evaluate(model,processor,judge,'test','v2')
    set_peft_model_state_dict(model,selected_state)
    del original,selected_state,optimizer,judge;gc.collect();torch.cuda.empty_cache()
    status('adapter_export');commit,hashes=export_reload(model,processor,latest,selected,selected['step'])
    del model,base,trainable;gc.collect();torch.cuda.empty_cache()
    status('fresh_download_reload',commit=commit)
    fresh=OUT/'redownload';assert not fresh.exists()
    snapshot_download(CONFIG['output_repo'],revision=commit,local_dir=fresh,allow_patterns=list(hashes))
    assert all(sha(fresh/name)==expected for name,expected in hashes.items())
    processor=AutoProcessor.from_pretrained(fresh,local_files_only=True)
    base=Gemma4ForConditionalGeneration.from_pretrained(CONFIG['base_model'],revision=CONFIG['base_revision'],
        dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa')
    restored=PeftModel.from_pretrained(base,fresh,local_files_only=True)
    saved=load_file(str(fresh/'adapter_model.safetensors'));actual=get_peft_model_state_dict(restored)
    assert set(saved)==set(actual) and all(torch.equal(saved[k].to(actual[k].dtype),actual[k].detach().cpu()) for k in saved)
    _,_,sample=generate(restored,processor,balanced_train()[0]);assert len(sample.strip())>20
    write('fresh_reload_verified.json',{'repo':CONFIG['output_repo'],'commit':commit,'hashes_verified':len(hashes),
        'local_files_only_adapter':True,'actual_adapter_tensor_match':True,'base_revision':CONFIG['base_revision'],
        'eligible_to_deploy':selected['eligible_to_deploy'],'raw_inference':sample,'format':'cumulative v2 plus RL PEFT adapter'})
    HfApi().upload_file(repo_id=CONFIG['output_repo'],path_or_fileobj=OUT/'fresh_reload_verified.json',
        path_in_repo='runs/'+CONFIG['run_id']+'/evidence/fresh_reload_verified.json')
    status('complete_training',commit=commit,eligible_to_deploy=selected['eligible_to_deploy'],steps=step,selected_step=selected['step'])

if __name__=='__main__':
    try:run()
    except BaseException as exc:
        write('failure.json',{'type':type(exc).__name__,'reason':str(exc)[:350],'time':time.time(),
            'traceback_frames':traceback.format_tb(exc.__traceback__)})
        status('failed',error_type=type(exc).__name__);raise SystemExit(1)
