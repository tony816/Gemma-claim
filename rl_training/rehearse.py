"""Real CPU mechanics, synthetic tiny weights; explicitly not a GPU quality gate."""
import inspect
import json
import tempfile
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, PeftModel, get_peft_model_state_dict
from transformers import Gemma4Config, Gemma4TextConfig, Gemma4ForConditionalGeneration, BitsAndBytesConfig

from .core import OUT, write, verify_release, advantages, completion_inputs, rollout_loss
from .guard import action


def run():
    torch.set_num_threads(2);torch.manual_seed(19)
    verify_release()
    text=Gemma4TextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
        global_head_dim=16,hidden_size_per_layer_input=0,vocab_size_per_layer_input=64,
        max_position_embeddings=128,sliding_window=16)
    config=Gemma4Config(text_config=text,image_token_id=60)
    base=Gemma4ForConditionalGeneration(config)
    original_base={k:v.clone() for k,v in base.state_dict().items()}
    model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    prompt={'input_ids':torch.tensor([[2,6,8,3]]),'attention_mask':torch.ones((1,4),dtype=torch.long),
            'mm_token_type_ids':torch.zeros((1,4),dtype=torch.long)}
    model.eval()
    with torch.inference_mode():sequence=model.generate(**prompt,max_new_tokens=4,do_sample=True)
    sequence=sequence.clone()
    inputs=completion_inputs(prompt,sequence);n=sequence.shape[-1]-4
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.train()
    before={k:v.clone() for k,v in get_peft_model_state_dict(model).items()}
    output=model(**inputs,use_cache=False,logits_to_keep=n+1)
    loss=rollout_loss(output.logits,sequence[:,-n:],advantages([.9,.1])[0]);loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters() if p.requires_grad)
    optimizer.step();after=get_peft_model_state_dict(model)
    assert any(not torch.equal(before[k],after[k]) for k in before)
    assert torch.equal(advantages([.5,.5]),torch.zeros(2))
    # The loss has no gradient on the final unused logit position.
    logits=torch.randn((1,3,64),requires_grad=True)
    rollout_loss(logits,torch.tensor([[3,4]]),1.).backward()
    assert logits.grad[:,-1].abs().sum()==0 and logits.grad[:,:-1].abs().sum()>0
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        torch.save(optimizer.state_dict(),Path(directory)/'optimizer.pt')
        restored_base=Gemma4ForConditionalGeneration(config);restored_base.load_state_dict(original_base)
        restored=PeftModel.from_pretrained(restored_base,directory,is_trainable=True)
        assert all(torch.equal(v,get_peft_model_state_dict(restored)[k]) for k,v in after.items())
        second=torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],lr=.01)
        second.load_state_dict(torch.load(Path(directory)/'optimizer.pt',weights_only=True))
        assert len(second.state)==len(optimizer.state)>0
        model.eval();restored.eval()
        with torch.inference_mode():
            assert torch.equal(model.generate(**prompt,max_new_tokens=3,do_sample=False),
                               restored.generate(**prompt,max_new_tokens=3,do_sample=False))
    # Positive/negative deadline checks do not call the provider.
    record={'deadline':1000}
    assert action(999,record,'online_rl',998)=='monitor'
    assert action(1000,record,'online_rl',999)=='interrupt_backup_delete'
    assert action(500,record,'failed',499)=='backup_and_delete'
    assert action(500,record,'complete_training',499)=='backup_and_delete'
    assert action(2000,{'deadline':5000},'online_rl',100)=='interrupt_backup_delete'
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    assert q.load_in_4bit
    assert 'logits_to_keep' in inspect.signature(Gemma4ForConditionalGeneration.forward).parameters
    judge=json.loads((OUT/'cpu_judge_encoding.json').read_text())
    assert judge['passed']==24
    write('cpu_rehearsal.json',{'passed':True,'frozen_hashes':283,'real_judge_encoded':24,
        'tiny_gemma_generated_completion_backward':True,'tiny_cumulative_save_reload_inference':True,
        'optimizer_resume':True,'deadline_failure_paths':5,'loss':float(loss.detach()),
        'skipped':['full-size GPU load','GPU backward','actual judge quality','provider termination mutation'],
        'prior_full_v2_meta_and_actual_tensor_check':'run_artifacts/rl_v3_20260908/cpu_preflight.json'})
    print('CPU REHEARSAL PASSED; full GPU and actual judge quality not yet tested')


if __name__=='__main__':run()
