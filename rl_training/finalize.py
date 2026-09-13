"""No-training recovery: finish fixed v2 comparison, export selected step 16, reload."""
import gc
import json
import shutil
import time
import traceback

import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoProcessor, Gemma4ForConditionalGeneration, set_seed

from .core import OUT, CONFIG, write, append, sha, rows, verify_release
from .train import status, generate
from .judge import Judge
from .followup import export_reload
from serving.rl_client import resolve_response


def remaining_v2_test(model, processor, judge):
    path = OUT/'evaluation/v2-test.jsonl'
    completed = [json.loads(line) for line in path.read_text().splitlines()]
    policies = rows('policy/test.jsonl')
    assert [r['episode_id'] for r in completed] == [p['episode_id'] for p in policies[:len(completed)]]
    assert len(completed) == 6
    for policy in policies[len(completed):]:
        _, sequence, raw = generate(model, processor, policy)
        write('pending_v2_generation.json', {'episode_id':policy['episode_id'], 'raw':raw,
            'v2_revision':CONFIG['v2_revision'], 'generation_max_tokens':768})
        judged = judge(policy, [raw], 'v2-test-'+policy['episode_id'])
        try:
            resolved = resolve_response(raw, policy, allow_code_fence=True, source_split='test')
            structure = {'valid':True, 'resolved':resolved}
        except (ValueError, TypeError) as exc:
            structure = {'valid':False, 'error_type':type(exc).__name__, 'error':str(exc)[:500]}
        item = {'episode_id':policy['episode_id'], 'task_type':policy['task_type'], 'raw':raw,
            'reward':judged['rewards'][0], 'judgment':judged['parsed']['candidates'][0], 'structure':structure}
        append('evaluation/v2-test.jsonl', item)
        completed.append(item)
        print('EVAL v2 test', len(completed), flush=True)
        del sequence
    summary = {}
    for task in ['all', 'claim_set', 'patent_advisory']:
        selected = [r for r in completed if task == 'all' or r['task_type'] == task]
        summary[task] = {'n':len(selected), 'mean_reward':sum(r['reward'] for r in selected)/len(selected),
            'severe_errors':sum(r['judgment']['severe_error'] for r in selected),
            'resolved_structure':sum(r['structure']['valid'] for r in selected)}
    write('evaluation/v2-test-summary.json', summary)


def run():
    set_seed(CONFIG['seed'])
    write('run_config.json', CONFIG)
    write('release_verified.json', verify_release())
    write('runtime.json', {'torch':torch.__version__, 'cuda':torch.version.cuda, 'gpu':torch.cuda.get_device_name(0)})
    shutil.copytree(OUT/'prior/evaluation', OUT/'evaluation')
    for name in ['selection.json', 'reference_unchanged.json', 'manual_test_review.json',
                 'manual_train10_review.json', 'evaluation_resolver_correction.json']:
        shutil.copyfile(OUT/'prior'/name, OUT/name)
    selected = json.loads((OUT/'selection.json').read_text())
    assert selected['step'] == 16 and selected['eligible_to_deploy']
    assert len((OUT/'evaluation/rl-test.jsonl').read_text().splitlines()) == 20
    write('recovery_scope.json', {'optimizer_updates':0, 'selected_step':16,
        'preserved_rl_test_outputs':20, 'preserved_v2_test_outputs':6,
        'only_remaining_v2_test_generated':14,
        'judge_change':'JSON string control-character escaping only; semantic rubric, weights and generation unchanged.',
        'model_selection_changed':False})
    status('judge_loading')
    judge = Judge()
    assert json.loads((OUT/'judge/configuration.json').read_text()) == json.loads((OUT/'prior/judge/configuration.json').read_text())
    status('v2_loading')
    processor = AutoProcessor.from_pretrained(CONFIG['base_model'], revision=CONFIG['base_revision'])
    base = Gemma4ForConditionalGeneration.from_pretrained(CONFIG['base_model'], revision=CONFIG['base_revision'],
        dtype=torch.bfloat16, device_map={'':'cuda:0'}, attn_implementation='sdpa')
    model = PeftModel.from_pretrained(base, CONFIG['v2_model'], revision=CONFIG['v2_revision'], is_trainable=False)
    original_file = hf_hub_download(CONFIG['v2_model'], 'adapter_model.safetensors', revision=CONFIG['v2_revision'])
    assert sha(original_file) == CONFIG['v2_weight_sha256']
    original = load_file(original_file)
    actual = get_peft_model_state_dict(model)
    assert set(original) == set(actual) and all(torch.equal(original[k].to(actual[k].dtype), actual[k].detach().cpu()) for k in original)
    del original, actual
    write('v2_loaded.json', {'revision':CONFIG['v2_revision'], 'exact_initial_tensor_match':True, 'training':False})
    status('remaining_v2_test')
    evaluation_complete = False
    try:
        remaining_v2_test(model, processor, judge)
        evaluation_complete = True
    except Exception as exc:
        write('evaluation_failure.json', {'type':type(exc).__name__, 'reason':str(exc)[:350],
            'frames':traceback.format_tb(exc.__traceback__), 'time':time.time()})
        # Even a comparison failure must not prevent preservation and actual
        # reload of the already selected, backed-up model. Deployment stays gated.
    del judge
    gc.collect(); torch.cuda.empty_cache()
    status('selected_checkpoint_loading')
    plan = CONFIG['resume_from']
    fresh_checkpoint = OUT/'resume_checkpoint'
    snapshot_download(plan['repo'], revision=plan['revision'], local_dir=fresh_checkpoint, allow_patterns=[plan['path']+'/*'])
    folder = fresh_checkpoint/plan['path']
    assert sha(folder/'hashes.json') == plan['manifest_sha256']
    manifest = json.loads((folder/'hashes.json').read_text())
    assert all(sha(folder/name) == digest for name, digest in manifest.items())
    assert sha(folder/'adapter_model.safetensors') == plan['adapter_sha256']
    weights = load_file(str(folder/'adapter_model.safetensors'))
    set_peft_model_state_dict(model, weights)
    actual = get_peft_model_state_dict(model)
    assert set(weights) == set(actual) and all(torch.equal(weights[k].to(actual[k].dtype), actual[k].detach().cpu()) for k in weights)
    del weights, actual
    write('selected_loaded.json', {**plan, 'exact_tensor_match':True, 'hashes_verified':len(manifest), 'optimizer_updates':0})
    effective = dict(selected)
    effective['eligible_to_deploy'] = selected['eligible_to_deploy'] and evaluation_complete
    status('adapter_export')
    commit, hashes = export_reload(model, processor, folder, effective, 16)
    del model, base
    gc.collect(); torch.cuda.empty_cache()
    status('fresh_download_reload', commit=commit)
    fresh = OUT/'redownload'
    assert not fresh.exists()
    snapshot_download(CONFIG['output_repo'], revision=commit, local_dir=fresh, allow_patterns=list(hashes))
    assert all(sha(fresh/name) == digest for name, digest in hashes.items())
    processor = AutoProcessor.from_pretrained(fresh, local_files_only=True)
    base = Gemma4ForConditionalGeneration.from_pretrained(CONFIG['base_model'], revision=CONFIG['base_revision'],
        dtype=torch.bfloat16, device_map={'':'cuda:0'}, attn_implementation='sdpa')
    restored = PeftModel.from_pretrained(base, fresh, local_files_only=True)
    saved = load_file(str(fresh/'adapter_model.safetensors'))
    actual = get_peft_model_state_dict(restored)
    assert set(saved) == set(actual) and all(torch.equal(saved[k].to(actual[k].dtype), actual[k].detach().cpu()) for k in saved)
    _, _, sample = generate(restored, processor, next(p for p in rows('policy/train.jsonl') if p['images']))
    assert len(sample.strip()) > 20
    write('fresh_reload_verified.json', {'repo':CONFIG['output_repo'], 'commit':commit,
        'hashes_verified':len(hashes), 'local_files_only_adapter':True, 'actual_adapter_tensor_match':True,
        'base_revision':CONFIG['base_revision'], 'eligible_to_deploy':effective['eligible_to_deploy'],
        'evaluation_complete':evaluation_complete, 'raw_inference':sample,
        'format':'cumulative v2 plus RL PEFT adapter', 'selected_step':16, 'recovery_optimizer_updates':0})
    HfApi().upload_file(repo_id=CONFIG['output_repo'], path_or_fileobj=OUT/'fresh_reload_verified.json',
        path_in_repo='runs/'+CONFIG['run_id']+'/evidence/fresh_reload_verified.json')
    status('complete_training', commit=commit, eligible_to_deploy=effective['eligible_to_deploy'],
        selected_step=16, recovery_optimizer_updates=0, evaluation_complete=evaluation_complete)


if __name__ == '__main__':
    try:
        run()
    except BaseException as exc:
        write('failure.json', {'type':type(exc).__name__, 'reason':str(exc)[:350],
            'traceback_frames':traceback.format_tb(exc.__traceback__), 'time':time.time()})
        status('failed', error_type=type(exc).__name__)
        raise SystemExit(1)
