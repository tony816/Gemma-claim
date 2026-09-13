"""Real tiny-Gemma continuation parity, plus actual checkpoint metadata checks."""
import tempfile,json
from pathlib import Path
import torch
from peft import LoraConfig,get_peft_model,PeftModel,get_peft_model_state_dict
from transformers import Gemma4Config,Gemma4TextConfig,Gemma4ForConditionalGeneration
from .core import OUT,ROOT,write
from .followup_math import activate,token_logps,objective
from .resume_state import restore
from .guard import missing_connection_action


def run():
    torch.set_num_threads(2);torch.manual_seed(18)
    cfg=Gemma4Config(text_config=Gemma4TextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=16,global_head_dim=16,
        hidden_size_per_layer_input=0,vocab_size_per_layer_input=64,max_position_embeddings=128,sliding_window=16),image_token_id=60)
    base=Gemma4ForConditionalGeneration(cfg);base_state={k:v.clone() for k,v in base.state_dict().items()}
    model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM',lora_dropout=0.))
    inputs={'input_ids':torch.tensor([[2,3,5,8,9,7]]),'attention_mask':torch.ones(1,6,dtype=torch.long)}
    def update(model,opt):
        opt.zero_grad(set_to_none=True);activate(model,'v2_reference');model.eval()
        with torch.no_grad():reference=token_logps(model(**inputs,logits_to_keep=3).logits,inputs['input_ids'][:,-2:])
        activate(model,'default');model.train()
        loss,_=objective(model(**inputs,logits_to_keep=3).logits,inputs['input_ids'][:,-2:],1.,reference,.02)
        loss.backward();opt.step()
    with tempfile.TemporaryDirectory() as d:
        d=Path(d);original=d/'v2';model.save_pretrained(original)
        model.load_adapter(original,adapter_name='v2_reference',is_trainable=False);activate(model,'default')
        reference_state={k:v.clone() for k,v in get_peft_model_state_dict(model,adapter_name='v2_reference').items()}
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
        update(model,opt)
        saved=d/'checkpoint';model.save_pretrained(saved,selected_adapters=['default'])
        torch.save({'optimizer':opt.state_dict(),'step':1,'episode':1,'torch_rng':torch.get_rng_state(),'cuda_rng':[]},saved/'resume.pt')
        expected_rng=torch.rand(4)
        update(model,opt);expected={k:v.clone() for k,v in get_peft_model_state_dict(model).items()}
        fresh_base=Gemma4ForConditionalGeneration(cfg);fresh_base.load_state_dict(base_state)
        fresh=PeftModel.from_pretrained(fresh_base,original,is_trainable=True)
        fresh.load_adapter(original,adapter_name='v2_reference',is_trainable=False);activate(fresh,'default')
        other=torch.optim.AdamW([p for p in fresh.parameters() if p.requires_grad],lr=.001)
        state=restore(fresh,other,saved);assert state['step']==1 and state['episode']==1
        assert torch.equal(torch.rand(4),expected_rng)
        update(fresh,other)
        assert all(torch.equal(v,get_peft_model_state_dict(fresh)[k]) for k,v in expected.items())
        assert all(torch.equal(v,get_peft_model_state_dict(fresh,adapter_name='v2_reference')[k]) for k,v in reference_state.items())
    actual=ROOT/'run_artifacts/rl_v3_followup/pod/checkpoints/step-0008/resume.pt'
    real=torch.load(actual,map_location='cpu',weights_only=True,mmap=True)
    assert real['step']==8 and real['episode']==8 and len(real['optimizer']['state'])==820
    record={'started_at':100,'deadline':10000}
    assert missing_connection_action(3279,record,True)=='retry'
    assert missing_connection_action(10000,record,True)=='stop_at_deadline'
    write('cpu_resume.json',{'passed':True,'tiny_gemma_exact_optimizer_continuation':True,
        'reference_unchanged':True,'torch_rng_exact_restoration':True,
        'actual_checkpoint_step':8,'actual_optimizer_parameter_states':820,
        'transient_missing_metadata_never_deletes_established_pod':True,
        'skipped':['full-size resumed GPU load','full-size CUDA RNG restoration']})
    print('RESUME CPU PASS: exact continued weights/optimizer/RNG, frozen reference, actual step-8 metadata, guard regression')

if __name__=='__main__':run()
