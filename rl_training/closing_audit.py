"""Read-only account/resource/lineage evidence; does not declare completion."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi
from tools.rl_cloud import credentials, runpod
from .core import ROOT, OUT, CONFIG, sha, verify_release, write
from .serve import snapshot


def run():
    credentials()
    with ThreadPoolExecutor(max_workers=4) as pool:
        queries = {
            'serverless': pool.submit(snapshot),
            'pods': pool.submit(runpod, '/pods', v1=True),
            'network_volumes': pool.submit(runpod, '/networkvolumes', v1=True),
            'billing_observed': pool.submit(runpod, '/billing?bucketSize=hour&lastN=12'),
        }
        result = {key: future.result() for key, future in queries.items()}
    api = HfApi()
    original = api.model_info(CONFIG['v2_model'])
    assert original.private and original.sha == CONFIG['v2_revision'], 'original_v2_changed'
    result['v2_preserved'] = {'repo': original.id, 'revision': original.sha, 'private': original.private}
    old_test = json.loads((OUT/'old_test_preserved.json').read_text(encoding='utf-8'))
    assert sha(Path(old_test['path'])) == old_test['sha256'], 'original_75_test_changed'
    result['original_75_test_preserved'] = old_test
    result['frozen_release_reverified'] = verify_release()
    proof_file = OUT/'pod/fresh_reload_verified.json'
    if proof_file.exists():
        proof = json.loads(proof_file.read_text(encoding='utf-8'))
        final = api.model_info(proof['repo'], revision=proof['commit'])
        assert final.private and final.sha == proof['commit']
        result['saved_model'] = {k: proof[k] for k in ('repo', 'commit', 'eligible_to_deploy', 'actual_adapter_tensor_match')}
    budget = json.loads((OUT/'additional_budget.json').read_text(encoding='utf-8'))
    account = result['serverless']['account']
    result['budget'] = {**budget, 'balance_now': account['clientBalance'],
        'observed_balance_debit_since_topup': budget['observed_balance_after_topup'] - account['clientBalance'],
        'note': 'Balance and billing are provider observations at query time; delayed accounting is possible.'}
    initial = json.loads((ROOT/'run_artifacts/rl_v3_20260908/account_gpu_prices.json').read_text(encoding='utf-8'))
    initial_balance = initial['data']['myself']['clientBalance']
    result['budget'].update(initial_observed_balance=initial_balance,
        known_deposit_during_task=budget['user_topup_usd'],
        total_observed_balance_debit=initial_balance+budget['user_topup_usd']-account['clientBalance'])
    result['training_resources'] = []
    for directory in ('rl_v3_runpod_only', 'rl_v3_runpod_schema', 'rl_v3_runpod_evidence', 'rl_v3_followup', 'rl_v3_resume', 'rl_v3_finalize'):
        folder = ROOT/'run_artifacts'/directory
        entry = {'artifacts': str(folder.relative_to(ROOT))}
        for filename in ('pod_created.json', 'termination.json', 'guard_finished.json', 'temporary_transfer.json'):
            path = folder/filename
            if path.exists(): entry[filename.removesuffix('.json')] = json.loads(path.read_text(encoding='utf-8'))
        training = folder/'pod/training.jsonl'
        if training.exists():
            records = [json.loads(line) for line in training.read_text(encoding='utf-8').splitlines()]
            entry['execution'] = {'episodes_logged': len(records),
                'optimizer_updates_executed': sum(row.get('gradient_norm', 0) > 0 for row in records),
                'highest_trajectory_step': max((row['step'] for row in records), default=0),
                'episode_seconds_sum': sum(row.get('seconds', 0) for row in records),
                'note': 'Executed operations include repeated updates after interruption; selected trajectory step is separate.'}
        result['training_resources'].append(entry)
    result['time'] = time.time()
    write('closing_audit.json', result)
    print(json.dumps({'time': result['time'], 'pods': len(result['pods']),
        'network_volumes': len(result['network_volumes']),
        'active_endpoint_pods': len(result['serverless']['active_pods']),
        'auto_pay': account['isAutoPayEnabled'], 'current_spend_per_hour': account['currentSpendPerHr'],
        'balance': account['clientBalance'], 'v2_unchanged': True}))


if __name__ == '__main__':
    run()
