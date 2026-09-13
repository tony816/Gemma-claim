"""Pod-resident cost guard: still works if the user's Windows PC sleeps."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import requests
from .core import OUT,CONFIG,write


def backup_small():
    from huggingface_hub import HfApi
    HfApi().upload_folder(repo_id=CONFIG['output_repo'],folder_path=OUT,path_in_repo='evidence',
        ignore_patterns=['model/*','checkpoints/*','redownload/*','adapter_export/*','*.log','*.env'])


def main():
    plan=json.loads((OUT/'adapter_recovery_plan.json').read_text())
    assert plan['pod_id']=='hff4ulh1b7ye73'
    assert plan['checkpoint_hf_download_verified'] is True
    # The cumulative checkpoint already exists in HF and as a hash-verified local copy.
    from huggingface_hub import HfApi
    assert HfApi().model_info(CONFIG['output_repo'],revision=plan['checkpoint_commit']).private
    while True:
        now=time.time()
        write('remote_guard_armed.json',{'time':now,'pid':os.getpid(),
            'pod_id':plan['pod_id'],'deadline':plan['hard_deadline'],
            'independent_of_user_pc':True})
        current=json.loads((OUT/'status.json').read_text())
        if current['stage'] in {'complete_training','failed'} or now>=plan['hard_deadline']-90:
            reason=current['stage'] if now<plan['hard_deadline']-90 else 'remote_hard_deadline'
            write('remote_guard_exit.json',{'time':now,'reason':reason,
                'checkpoint_commit_preserved':plan['checkpoint_commit']})
            try:
                subprocess.run([sys.executable,'-m','rl_training.remote_guard','backup'],timeout=60,
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
            except Exception:pass  # The exact cumulative checkpoint is already verified off-Pod.
            if reason in {'complete_training','failed'}:
                time.sleep(30)  # Let the local guard copy receipts first when the PC is awake.
            if reason=='remote_hard_deadline':
                while time.time()<plan['hard_deadline']:time.sleep(1)
            url='https://rest.runpod.io/v1/pods/'+plan['pod_id']
            headers={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY']}
            while True:
                try:
                    result=requests.delete(url,headers=headers,timeout=20)
                    if result.status_code in {200,204,404}:return
                except requests.RequestException:pass
                time.sleep(5)
        time.sleep(10)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='backup':backup_small()
    else:main()
