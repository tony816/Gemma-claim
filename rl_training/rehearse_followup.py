"""Check the changed reward, actual tiny Gemma gradients, and frozen reference."""
import json
import tempfile
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, PeftModel
from transformers import Gemma4Config, Gemma4TextConfig, Gemma4ForConditionalGeneration
from .core import OUT, write, sha, completion_inputs
from .followup_math import rewards_for, weighted_advantages, token_logps, objective, activate


def run():
    packets=json.loads((OUT/'prior/judge/packets.json').read_text())
    labels={x['id']:x for x in json.loads((OUT/'prior/judge/labels.PRIVATE.json').read_text())}
    responses={x['id']:x for x in map(json.loads,(OUT/'prior/judge/responses.jsonl').read_text().splitlines())}
    results=[]
    for packet in packets:
        actual=responses[packet['id']]['parsed']
        rewards,errors=rewards_for(packet['policy'],packet['candidates'],actual)
        preference='A' if rewards[0]>rewards[1] else 'B' if rewards[1]>rewards[0] else 'tie'
        results.append({'id':packet['id'],'rewards':rewards,'errors':errors,
                        'correct':preference==labels[packet['id']]['expected']})
    assert sum(x['correct'] for x in results)==24
    assert weighted_advantages([1.,.99]).abs().max()<.051
    assert weighted_advantages([1.,.7]).abs().min()>.99
    assert weighted_advantages([.25,.25]).abs().max()==0
    torch.set_num_threads(2);torch.manual_seed(19)
    text=Gemma4TextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,head_dim=16,global_head_dim=16,
        hidden_size_per_layer_input=0,vocab_size_per_layer_input=64,max_position_embeddings=128,sliding_window=16)
    config=Gemma4Config(text_config=text,image_token_id=60)
    base=Gemma4ForConditionalGeneration(config)
    base_state={k:v.clone() for k,v in base.state_dict().items()}
    model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        model.load_adapter(directory,adapter_name='v2_reference',is_trainable=False)
        reference={k:v.clone() for k,v in get_peft_model_state_dict(model,adapter_name='v2_reference').items()}
        activate(model,'default')
        parameters=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=.01)
        prompt={'input_ids':torch.tensor([[2,6,8,3]]),'attention_mask':torch.ones((1,4),dtype=torch.long)}
        model.eval()
        with torch.inference_mode():seq=model.generate(**prompt,max_new_tokens=4,do_sample=True)
        seq=seq.clone();inputs=completion_inputs(prompt,seq);n=seq.shape[1]-4
        activate(model,'v2_reference');model.eval()
        with torch.no_grad():ref=token_logps(model(**inputs,logits_to_keep=n+1).logits,seq[:,-n:])
        activate(model,'default');model.train()
        out=model(**inputs,logits_to_keep=n+1)
        loss,kl=objective(out.logits,seq[:,-n:],1.,ref,.02)
        loss.backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in parameters)
        optimizer.step()
        assert all(torch.equal(v,get_peft_model_state_dict(model,adapter_name='v2_reference')[k]) for k,v in reference.items())
        trained=get_peft_model_state_dict(model)
        assert any(not torch.equal(trained[k],reference[k]) for k in trained)
        model.save_pretrained(Path(directory)/'new',selected_adapters=['default'])
        fresh_base=Gemma4ForConditionalGeneration(config);fresh_base.load_state_dict(base_state)
        fresh=PeftModel.from_pretrained(fresh_base,Path(directory)/'new')
        model.eval();fresh.eval()
        with torch.inference_mode():
            assert torch.equal(model.generate(**prompt,max_new_tokens=3),fresh.generate(**prompt,max_new_tokens=3))
    write('cpu_followup.json',{'passed':True,'compound_calibration_agreement':24,'total':24,
        'provenance':'CPU rescore of actual frozen RunPod judge outputs, unchanged model/system/runtime; not new GPU inference',
        'results':results,'frozen_reference_unchanged':True,'tiny_actual_gemma_gradient':True,
        'cumulative_adapter_fresh_reload_inference':True,'loss':float(loss.detach()),'initial_kl':float(kl.detach()),
        'skipped':['full-size GPU followup','new model quality']})
    print('FOLLOWUP CPU PASS: 24/24 scalar agreement, tiny Gemma KL/reference/save/reload')

if __name__=='__main__':run()
