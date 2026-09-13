"""Read-only frozen-material and actual v2 adapter checks before rented compute.

This is deliberately a CPU gate, never evidence of GPU or RL execution.
"""
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.rl_cloud import credentials, OUT, save
from tools.rl_judge import RELEASE, rows

BASE = 'google/gemma-4-31B-it'
BASE_SHA = '842da3794eaa0b77d5f08bae87a17459d91ff475'
V2 = 'Mepeng22/gemma-4-31b-claim-lora-v2'
V2_SHA = 'cc04579366bbb88663029d53b2aafcc84159589e'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def policy_messages(row):
    """Only messages/images cross this boundary; reward files are never read."""
    if set(row) != {'episode_id', 'task_type', 'images', 'messages'}:
        raise ValueError('unexpected_policy_fields')
    messages = []
    for message in row['messages']:
        if message['role'] not in {'system', 'user'}:
            raise ValueError('policy_contains_assistant_or_unexpected_role')
        content = message['content']
        if not isinstance(content, str):
            raise ValueError('unexpected_release_message_shape')
        messages.append({'role': message['role'], 'content': [{'type': 'text', 'text': content}]})
    if row['images']:
        user = next(m for m in messages if m['role'] == 'user')
        user['content'] = [{'type': 'image'} for _ in row['images']] + user['content']
    return messages


def encode_policy(processor, row):
    from PIL import Image
    messages = policy_messages(row)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = []
    for rel in row['images']:
        path = (RELEASE/rel).resolve()
        if not path.is_relative_to(RELEASE.resolve()):
            raise ValueError('image_outside_release')
        with Image.open(path) as image:
            images.append(image.convert('RGB').copy())
    kwargs = {'text': text, 'return_tensors': 'pt', 'padding': False}
    if images:
        kwargs['images'] = images
    return processor(**kwargs)


def main():
    import os
    credentials()
    import torch
    from accelerate import init_empty_weights
    from huggingface_hub import HfApi, hf_hub_download
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from safetensors import safe_open
    from transformers import AutoConfig, AutoProcessor, Gemma4ForConditionalGeneration

    torch.set_num_threads(2)
    receipt = json.loads((ROOT/'.superloopy/evidence/frozen-release.json').read_text())
    manifest = json.loads((RELEASE/'manifest.json').read_text())
    assert sha(RELEASE/'manifest.json') == receipt['manifest_sha256'], 'receipt mismatch'
    assert all(sha(RELEASE/path) == expected for path, expected in manifest['files'].items()), 'file mismatch'
    api = HfApi(token=os.environ['HF_TOKEN'])
    info = api.model_info(V2, revision=V2_SHA, files_metadata=True)
    assert info.sha == V2_SHA and info.private
    config_path = hf_hub_download(V2, 'adapter_config.json', revision=V2_SHA, token=os.environ['HF_TOKEN'])
    weights = hf_hub_download(V2, 'adapter_model.safetensors', revision=V2_SHA, token=os.environ['HF_TOKEN'])
    remote_weight = next(x for x in info.siblings if x.rfilename == 'adapter_model.safetensors')
    weight_hash = sha(weights)
    assert remote_weight.lfs.sha256 == weight_hash, 'v2 actual weight hash mismatch'
    adapter_config = json.loads(Path(config_path).read_text())
    assert adapter_config['base_model_name_or_path'] == BASE
    processor = AutoProcessor.from_pretrained(BASE, revision=BASE_SHA, token=os.environ['HF_TOKEN'])
    base_config = AutoConfig.from_pretrained(BASE, revision=BASE_SHA, token=os.environ['HF_TOKEN'])
    lora_config = LoraConfig.from_pretrained(str(Path(config_path).parent))
    lora_config.inference_mode = False
    with init_empty_weights():
        skeleton = Gemma4ForConditionalGeneration(base_config)
        skeleton = get_peft_model(skeleton, lora_config)
    expected = {k: tuple(v.shape) for k, v in get_peft_model_state_dict(skeleton).items()}
    with safe_open(weights, framework='pt', device='cpu') as tensors:
        actual = {k: tuple(tensors.get_slice(k).get_shape()) for k in tensors.keys()}
        assert actual == expected, 'v2 adapter shape/name incompatibility'
        nonzero, finite = 0, True
        for name in tensors.keys():
            tensor = tensors.get_tensor(name)
            finite = finite and bool(torch.isfinite(tensor).all())
            nonzero += int(torch.count_nonzero(tensor))
        assert finite and nonzero > 0
    trainable = sum(p.numel() for p in skeleton.parameters() if p.requires_grad)
    assert trainable == 122429440, 'unexpected v2 trainable parameter count'
    assert all(not any(x in n for x in ['vision_tower', 'audio_tower'])
               for n, p in skeleton.named_parameters() if p.requires_grad)
    forward = inspect.signature(Gemma4ForConditionalGeneration.forward)
    assert 'logits_to_keep' in forward.parameters
    del skeleton
    encoded = []
    ids = set()
    for split, count in [('train', 88), ('calibration', 20), ('test', 20)]:
        policies = rows(RELEASE/f'policy/{split}.jsonl')
        assert len(policies) == count
        for policy in policies:
            assert policy['episode_id'] not in ids, 'overlapping episode id'
            ids.add(policy['episode_id'])
            tokens = encode_policy(processor, policy)
            length = int(tokens['input_ids'].shape[-1])
            assert 0 < length <= 8192, 'input over planned token limit'
            encoded.append({'split': split, 'episode_id': policy['episode_id'],
                            'images': len(policy['images']), 'prompt_tokens': length,
                            'tensor_keys': sorted(tokens)})
        print(split, len(policies), 'CPU policy encoding passed', flush=True)
    report = {'passed': True, 'device': 'cpu', 'files_verified': len(manifest['files']),
        'manifest_sha256': receipt['manifest_sha256'], 'v2_repo': V2, 'v2_revision': V2_SHA,
        'v2_adapter_sha256': weight_hash, 'adapter_tensors': len(expected),
        'adapter_all_finite': finite, 'adapter_nonzero_elements': nonzero,
        'trainable_parameters_from_v2_config': trainable,
        'base_repo': BASE, 'base_revision': BASE_SHA,
        'historical_base_weight_revision_recovered': False,
        'base_revision_limit': 'Historical v2 run recorded main only. This is a current pinned compatible config; GPU weight loading remains unverified.',
        'policy_only_encoding': encoded, 'reward_answers_in_policy': False,
        'base_weights_loaded': False, 'v2_weights_read_on_cpu': True,
        'gpu_runtime_verified': False, 'online_rl_executed': False,
        'judge_calibration_passed': False, 'gpu_allowed_by_this_report': False}
    save('cpu_preflight.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'policy_only_encoding'}, indent=2))


if __name__ == '__main__':
    main()
