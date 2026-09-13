"""Restore cumulative policy, optimizer, and RNG from a hash-verified checkpoint."""
import json
from pathlib import Path
import torch
from peft import get_peft_model_state_dict,set_peft_model_state_dict
from safetensors.torch import load_file
from .core import CONFIG,OUT,sha,write


def restore(model,optimizer,folder):
    folder=Path(folder)
    saved=load_file(str(folder/'adapter_model.safetensors'))
    set_peft_model_state_dict(model,saved,adapter_name='default')
    actual=get_peft_model_state_dict(model,adapter_name='default')
    assert set(saved)==set(actual)
    assert all(torch.equal(saved[k].to(actual[k].dtype),actual[k].detach().cpu()) for k in saved)
    state=torch.load(folder/'resume.pt',map_location='cpu',weights_only=True)
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    if torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda_rng'])
    return {'step':state['step'],'episode':state['episode'],'adapter_tensor_match':True,
        'optimizer_states':len(optimizer.state),'torch_rng_restored':True,'cuda_rng_restored':torch.cuda.is_available()}


def from_hub(model,optimizer):
    from huggingface_hub import snapshot_download,HfApi
    plan=CONFIG['resume_from'];folder=OUT/'resume_checkpoint'
    assert HfApi().model_info(plan['repo'],revision=plan['revision']).private
    snapshot_download(plan['repo'],revision=plan['revision'],local_dir=folder,allow_patterns=[plan['path']+'/*'])
    checkpoint=folder/plan['path']
    assert sha(checkpoint/'hashes.json')==plan['manifest_sha256']
    hashes=json.loads((checkpoint/'hashes.json').read_text())
    assert all(sha(checkpoint/name)==digest for name,digest in hashes.items())
    assert sha(checkpoint/'adapter_model.safetensors')==plan['adapter_sha256']
    result=restore(model,optimizer,checkpoint)
    assert result['step']==plan['step'] and result['episode']==plan['episode']
    write('resume_loaded.json',{**plan,**result,'fresh_hub_download_hashes_verified':len(hashes)})
    return result
