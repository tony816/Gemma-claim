"""External Windows watchdog, independent of the chat and GPU training process."""
import json
import os
import time

from .core import OUT, CONFIG, write, append
from .cloud import connect, command, sync, backup_verified, terminate
from tools.rl_cloud import runpod


def action(now, record, stage, last_progress):
    if stage in {'failed','complete_training'}:return 'backup_and_delete'
    if now>=record['deadline']:return 'interrupt_backup_delete'
    if now-last_progress>1200:return 'interrupt_backup_delete'
    return 'monitor'


def missing_connection_action(now, record, ever_provisioned):
    # REST can transiently omit publicIp/portMappings for an established Pod.
    # Missing connection metadata is never evidence that its payload is absent.
    if ever_provisioned:
        return 'stop_at_deadline' if now>=record['deadline'] else 'retry'
    return 'delete_unprovisioned' if now-record['started_at']>900 else 'retry'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    while True:
        write('guard_armed.json',{'pid':os.getpid(),'heartbeat':time.time(),
            'hard_seconds':CONFIG['hard_pod_deadline_seconds'],'independent_of_chat':True})
        if (OUT/'termination.json').exists():break
        path=OUT/'pod_created.json'
        if not path.exists():time.sleep(10);continue
        record=json.loads(path.read_text())
        try:
            pods=runpod('/pods',v1=True)
            pod=next((p for p in pods if p['id']==record['id']),None)
            if pod is None:
                write('guard_finished.json',{'time':time.time(),'pod_absent':True});break
            if not pod.get('publicIp') or not (pod.get('portMappings') or {}).get('22'):
                ever_provisioned=(OUT/'provisioned.json').exists() or (OUT/'pod/status.json').exists()
                decision=missing_connection_action(time.time(),record,ever_provisioned)
                if decision=='delete_unprovisioned':
                    terminate(record['id'],'startup_never_ready_no_payload');break
                if decision=='stop_at_deadline':
                    runpod('/pods/'+record['id']+'/stop','POST',v1=True)
                    write('guard_needs_recovery.json',{'time':time.time(),'pod_id':record['id'],
                        'reason':'deadline_with_transient_missing_connection_metadata'})
                    break
                time.sleep(20);continue
            with connect(pod) as client:
                sync(client,include_checkpoints=False)
                status_path=OUT/'pod/status.json'
                current=json.loads(status_path.read_text()) if status_path.exists() else {}
                stage=current.get('stage')
                last_progress=current.get('time',record['started_at'])
                code,result,_=command(client,'stat -c %Y /workspace/boot.log 2>/dev/null || true')
                if result.strip().isdigit():last_progress=max(last_progress,float(result.strip()))
                decision=action(time.time(),record,stage,last_progress)
                plan=OUT/'repair_plan.json'
                failure=OUT/'pod/failure.json'
                if stage=='failed' and plan.exists() and failure.exists() and time.time()<record['deadline']:
                    reason=json.loads(failure.read_text()).get('reason')
                    if reason=='judge_calibration_below_85_percent':
                        hold=OUT/'repair_hold_started.json'
                        if not hold.exists():write('repair_hold_started.json',{'time':time.time()})
                        if time.time()-json.loads(hold.read_text())['time']<120:
                            decision='monitor'
                write('guard_status.json',{'time':time.time(),'stage':stage,'decision':decision,
                    'elapsed_seconds':time.time()-record['started_at'],'last_progress':last_progress})
                if decision=='monitor':time.sleep(20);continue
                if decision=='interrupt_backup_delete':
                    module=CONFIG.get('training_module','rl_training.train')
                    assert module in {'rl_training.train','rl_training.followup','rl_training.finalize'}
                    command(client,"pkill -TERM -f '^/workspace/rl-venv/bin/python -u -m "+module+"$' || true")
                    time.sleep(3)
                sync(client,include_checkpoints=True)
                # Full copied files plus hash manifests; success must include actual HF reload proof.
                backup=backup_verified()
                if stage=='complete_training':
                    assert (OUT/'pod/fresh_reload_verified.json').exists()
                if backup['verified']:
                    terminate(record['id'],decision+':'+str(stage));break
                # Stop GPU immediately when safe export is incomplete; keep only recoverable
                # disk temporarily. It is never reported as terminated or completed.
                runpod('/pods/'+record['id']+'/stop','POST',v1=True)
                write('guard_needs_recovery.json',{'time':time.time(),'pod_id':record['id'],'backup':backup})
                break
        except Exception as exc:
            append('guard_errors.jsonl',{'time':time.time(),'type':type(exc).__name__})
            if time.time()>record['deadline']:
                # Provider stop remains possible even when SSH fails. Protect GPU budget;
                # do not delete unique data that could not be exported.
                try:runpod('/pods/'+record['id']+'/stop','POST',v1=True)
                except Exception:pass
        time.sleep(20)


if __name__=='__main__':main()
