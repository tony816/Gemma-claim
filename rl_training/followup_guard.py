"""Boot-started independent deadline guard. Never deletes an unbacked checkpoint."""
import json, os, signal, subprocess, sys, time
from pathlib import Path
import requests
from .core import OUT, CONFIG, write


def close_action(stage, checkpoint_exists, backup_verified):
    if stage=='complete_training' or not checkpoint_exists or backup_verified:return 'delete'
    return 'stop'


def main():
    pod=os.environ['RL_POD_ID'];deadline=float(os.environ['RL_STARTED_AT'])+CONFIG['hard_pod_deadline_seconds']
    headers={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY']}
    while True:
        now=time.time()
        write('remote_guard_armed.json',{'time':now,'pid':os.getpid(),'pod_id':pod,
            'deadline':deadline,'independent_of_user_pc':True})
        try:current=json.loads((OUT/'status.json').read_text())
        except (OSError,ValueError):current={}
        stage=current.get('stage')
        if stage in {'failed','complete_training'} or now>=deadline-240:
            if stage not in {'failed','complete_training'}:
                module=CONFIG.get('training_module','rl_training.followup')
                assert module in {'rl_training.train','rl_training.followup','rl_training.finalize'}
                result=subprocess.run(['pgrep','-f','^/workspace/rl-venv/bin/python -u -m '+module+'$'],capture_output=True,text=True)
                for pid in result.stdout.split():
                    try:os.kill(int(pid),signal.SIGTERM)
                    except ProcessLookupError:pass
            checkpoint_dirs=sorted((OUT/'checkpoints').glob('step-*'))
            latest={}
            try:latest=json.loads((OUT/'latest_checkpoint.json').read_text())
            except (OSError,ValueError):pass
            backed=bool(latest.get('commit') and checkpoint_dirs and Path(latest['path']).name==checkpoint_dirs[-1].name)
            action=close_action(stage,bool(checkpoint_dirs),backed)
            if stage=='complete_training' and not (OUT/'fresh_reload_verified.json').exists():action='stop'
            write('remote_guard_exit.json',{'time':time.time(),'reason':stage or 'hard_deadline',
                'action':action,'checkpoint_commit':latest.get('commit'),'independent_of_user_pc':True})
            python='/workspace/rl-venv/bin/python' if Path('/workspace/rl-venv/bin/python').exists() else sys.executable
            try:subprocess.run([python,'-m','rl_training.followup_guard','backup'],timeout=90,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
            except Exception:pass
            # A local guard can collect small receipts during this grace period.
            time.sleep(min(30,max(0,deadline-time.time()-25)))
            url='https://rest.runpod.io/v1/pods/'+pod
            while True:
                try:
                    result=(requests.delete(url,headers=headers,timeout=20) if action=='delete'
                        else requests.post(url+'/stop',headers=headers,timeout=20))
                    if result.status_code in {200,204,404}:return
                except requests.RequestException:pass
                time.sleep(5)
        time.sleep(10)

if __name__=='__main__':
    if len(sys.argv)>1:
        from huggingface_hub import HfApi
        HfApi().upload_folder(repo_id=CONFIG['output_repo'],folder_path=OUT,
            path_in_repo='runs/'+CONFIG['run_id']+'/evidence',
            ignore_patterns=['checkpoints/*','resume_checkpoint/*','redownload/*','adapter_export/*','*.log','*.env'])
    else:main()
