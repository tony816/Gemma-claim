"""Private local control plane, SSH and recoverable copies, no token output."""
import json
import os
from pathlib import Path
import posixpath
import tarfile
import time

import paramiko
from tools.rl_cloud import runpod, graphql, credentials
from .core import ROOT, OUT, CONFIG, sha, write

REMOTE='/workspace/Gemma-claim'
REMOTE_OUT=REMOTE+'/'+CONFIG['artifact_directory']


def connect(pod):
    client=paramiko.SSHClient()
    known=OUT/'ssh_known_hosts'
    if known.exists():client.load_host_keys(str(known))
    # Fresh provider-assigned IP: trust on first use, persist fingerprint for later connections.
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(pod['publicIp'],port=int(pod['portMappings']['22']),username='root',
        key_filename=str(Path.home()/'.ssh/runpod_gemma_ed25519'),timeout=20,
        banner_timeout=20,auth_timeout=20,look_for_keys=False,allow_agent=False)
    client.save_host_keys(str(known))
    return client


def command(client, cmd, timeout=30):
    _,out,err=client.exec_command(cmd,timeout=timeout)
    result=out.read().decode('utf-8','replace');error=err.read().decode('utf-8','replace')
    code=out.channel.recv_exit_status()
    return code,result,error


def bundle():
    destination=OUT/'payload.tar.gz'
    includes=['rl_training','tools/rl_judge.py','tools/rl_cloud.py','serving/rl_client.py',
              'rl_materials/contracts.py','.superloopy/evidence/frozen-release.json',CONFIG['release']]
    if CONFIG.get('training_module') in {'rl_training.followup','rl_training.finalize'}:
        includes += [CONFIG['artifact_directory']+'/prior',CONFIG['artifact_directory']+'/repo_owned.json',
                     CONFIG['artifact_directory']+'/precommit.json']
    with tarfile.open(destination,'w:gz') as archive:
        for name in includes:
            path=ROOT/name
            files=[path] if path.is_file() else path.rglob('*')
            for file in files:
                if file.is_file() and '__pycache__' not in file.parts and file.suffix not in {'.pyc','.env'}:
                    archive.add(file,arcname=file.relative_to(ROOT).as_posix())
    return destination


def sync(client, include_checkpoints=True):
    sftp=paramiko.SFTPClient.from_transport(client.get_transport(),window_size=64*1024*1024);downloaded=[]
    def walk(remote, local):
        import stat
        local.mkdir(parents=True,exist_ok=True)
        entries=sftp.listdir_attr(remote)
        # The Hub upload may fail before latest_checkpoint.json is advanced.
        # Preserve the newest ON-DISK step, including a partial latest write;
        # an incomplete hash manifest then prevents deletion instead of losing it.
        if remote.endswith('/checkpoints'):
            names=[e.filename for e in entries if e.filename.startswith('step-')]
            if names:entries=[e for e in entries if e.filename==max(names)]
        for entry in entries:
            if entry.filename in {'model','redownload','adapter_export','resume_checkpoint','__pycache__','.cache'}:continue
            if not include_checkpoints and entry.filename=='checkpoints':continue
            if entry.filename.endswith(('.env','.tmp')):continue
            src=posixpath.join(remote,entry.filename);dst=local/entry.filename
            if stat.S_ISDIR(entry.st_mode):walk(src,dst)
            else:
                if dst.exists() and dst.stat().st_size==entry.st_size and dst.stat().st_mtime>=entry.st_mtime:continue
                tmp=dst.with_suffix(dst.suffix+'.download')
                sftp.get(src,str(tmp));tmp.replace(dst)
                os.utime(dst,(entry.st_mtime,entry.st_mtime));downloaded.append(str(dst.relative_to(OUT)))
    try:walk(REMOTE_OUT,OUT/'pod')
    except FileNotFoundError:pass
    finally:sftp.close()
    return downloaded


def backup_verified():
    root=OUT/'pod/checkpoints'
    if not root.exists():return {'checkpoints':0,'verified':True}
    checkpoints=[]
    for directory in root.glob('step-*'):
        manifest=directory/'hashes.json'
        if not manifest.exists():return {'verified':False,'reason':'partial_checkpoint'}
        expected=json.loads(manifest.read_text())
        if not all((directory/name).exists() and sha(directory/name)==value for name,value in expected.items()):
            return {'verified':False,'reason':'checkpoint_hash_mismatch'}
        checkpoints.append(directory.name)
    return {'checkpoints':len(checkpoints),'verified':True,'names':checkpoints}


def terminate(pod_id, reason):
    runpod('/pods/'+pod_id,'DELETE',v1=True)
    remaining=runpod('/pods',v1=True)
    assert all(p['id']!=pod_id for p in remaining)
    write('termination.json',{'pod_id':pod_id,'reason':reason,'time':time.time(),
        'pod_absent':True,'local_backup':backup_verified()})


def safe_finish(pod_id, reason):
    pod=runpod('/pods/'+pod_id,v1=True)
    with connect(pod) as client:
        sync(client)
    backup=backup_verified()
    assert backup['verified'],'backup_not_verified'
    terminate(pod_id,reason)


def launch():
    credentials();OUT.mkdir(parents=True,exist_ok=True)
    assert (OUT/'cpu_rehearsal.json').exists()
    assert json.loads((OUT/'cpu_rehearsal.json').read_text())['passed']
    assert json.loads((OUT/'cpu_schema.json').read_text())['passed']
    if CONFIG.get('training_module')=='rl_training.followup':
        assert json.loads((OUT/'cpu_followup.json').read_text())['passed']
        assert json.loads((OUT/'precommit.json').read_text())['maximum_additional_usd']==12
    if CONFIG.get('training_module')=='rl_training.finalize':
        assert json.loads((OUT/'cpu_finalize.json').read_text())['passed']
    assert (OUT/'guard_armed.json').exists()
    guard=json.loads((OUT/'guard_armed.json').read_text())
    assert time.time()-guard['heartbeat']<90, 'external_guard_not_live'
    assert not runpod('/pods',v1=True),'unexpected_existing_pods'
    assert not (OUT/'pod_created.json').exists(),'refuse_duplicate_launch'
    before=graphql('query { myself { currentSpendPerHr clientBalance } }')
    if CONFIG.get('resume_from'):
        budget=json.loads((OUT/'additional_budget.json').read_text())
        spent=budget['observed_balance_after_topup']-before['data']['myself']['clientBalance']
        assert spent+CONFIG['hard_pod_deadline_seconds']/3600*CONFIG.get('conservative_pod_usd_hour',3.746)+1.0<=12,'resume_exceeds_remaining_budget'
        assert json.loads((OUT/'cpu_resume.json').read_text())['passed']
    write('account_before.json',before)
    body={'name':'gemma-v2-online-rl-20260908','imageName':'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404',
        'cloudType':CONFIG.get('pod_cloud_type','COMMUNITY'),'gpuTypeIds':[CONFIG['pod_gpu']],'gpuCount':1,
        'containerDiskInGb':30,'volumeInGb':CONFIG.get('pod_volume_gb',300),'volumeMountPath':'/workspace',
        'ports':['22/tcp'],'env':{'PUBLIC_KEY':(Path.home()/'.ssh/runpod_gemma_ed25519.pub').read_text().strip()},
        'supportPublicIp':True}
    started=time.time()
    write('launch_attempt.json', {'time':started, 'gpu':CONFIG['pod_gpu'],
        'hard_seconds':CONFIG['hard_pod_deadline_seconds'], 'resources_absent_before_request':True})
    pod=runpod('/pods','POST',body,v1=True)
    record={k:pod.get(k) for k in ['id','name','costPerHr','desiredStatus','publicIp','portMappings']}
    record.update(started_at=started,deadline=started+CONFIG['hard_pod_deadline_seconds'])
    write('pod_created.json',record)
    print(json.dumps(record),flush=True)
    if record.get('costPerHr',0)>CONFIG.get('max_gpu_usd_hour',3.70):
        terminate(pod['id'],'actual_price_above_cap');raise RuntimeError('pod_actual_price_above_cap')


def provision():
    credentials();record=json.loads((OUT/'pod_created.json').read_text())
    pod=runpod('/pods/'+record['id'],v1=True)
    with connect(pod) as client:
        archive=bundle()
        # Credentials travel on an encrypted stream, never shell command arguments or pod metadata.
        import shlex
        values={'HF_TOKEN':os.environ['HF_TOKEN'],'RUNPOD_API_KEY':os.environ['RUNPOD_API_KEY'],
                'RL_STARTED_AT':str(record['started_at']),'RL_POD_ID':record['id'],
                'RL_OUT':REMOTE_OUT,'RL_TRAINING_MODULE':CONFIG.get('training_module','rl_training.train')}
        stdin,stdout,stderr=client.exec_command('umask 077; cat > /workspace/pod.env; chmod 600 /workspace/pod.env')
        stdin.write(''.join(k+'='+shlex.quote(v)+'\n' for k,v in values.items()));stdin.channel.shutdown_write()
        assert stdout.channel.recv_exit_status()==0
        # HF is file transport only, never training infrastructure. The temporary
        # PRIVATE dataset avoids a measured ~7 minute SSH transfer bottleneck.
        from huggingface_hub import HfApi
        api=HfApi();repo='Mepeng22/gemma-claim-rl-transfer-'+record['id']
        api.create_repo(repo,repo_type='dataset',private=True,exist_ok=False)
        transfer={'repo':repo,'type':'dataset','private':True,'created_at':time.time(),
            'payload_sha256':sha(archive),'purpose':'temporary encrypted file transport, no HF compute'}
        write('temporary_transfer.json',transfer)
        assert api.repo_info(repo,repo_type='dataset').private
        commit=api.upload_file(repo_id=repo,repo_type='dataset',path_or_fileobj=archive,
                              path_in_repo='payload.tar.gz',commit_message='Temporary frozen RunPod payload')
        transfer['commit']=commit.oid;write('temporary_transfer.json',transfer)
        bootstrap=('import os,requests,hashlib\n'
            +f'url="https://huggingface.co/datasets/{repo}/resolve/{commit.oid}/payload.tar.gz"\n'
            +'r=requests.get(url,headers={"Authorization":"Bearer "+os.environ["HF_TOKEN"]},stream=True,timeout=120)\n'
            +'assert r.status_code==200, r.status_code\n'
            +'with open("/workspace/payload.tar.gz","wb") as f:\n'
            +' for chunk in r.iter_content(8*1024*1024): f.write(chunk)\n'
            +'with open("/workspace/payload.tar.gz","rb") as f:\n'
            +f' assert hashlib.file_digest(f,"sha256").hexdigest()=="{transfer["payload_sha256"]}"\n'
            +'print("payload hash verified")\n')
        stdin,stdout,stderr=client.exec_command("bash -c 'set -a; source /workspace/pod.env; set +a; python -'",timeout=150)
        stdin.write(bootstrap);stdin.channel.shutdown_write()
        assert stdout.channel.recv_exit_status()==0,'private_transfer_failed'
        code,_,_=command(client,'mkdir -p /workspace/Gemma-claim && tar -xzf /workspace/payload.tar.gz -C /workspace/Gemma-claim')
        assert code==0
        api.delete_repo(repo,repo_type='dataset')
        transfer['deleted_at']=time.time();transfer['remote_payload_hash_verified']=True
        write('temporary_transfer.json',transfer)
        code,out,err=command(client,'nohup bash /workspace/Gemma-claim/rl_training/boot.sh > /workspace/boot.log 2>&1 < /dev/null &',timeout=10)
        write('provisioned.json',{'time':time.time(),'payload_sha256':sha(archive),'pod_id':record['id']})


if __name__=='__main__':
    import sys
    if sys.argv[1]=='launch':launch()
    elif sys.argv[1]=='provision':provision()
    elif sys.argv[1]=='sync':
        record=json.loads((OUT/'pod_created.json').read_text())
        with connect(runpod('/pods/'+record['id'],v1=True)) as client:
            print(sync(client,include_checkpoints=False))
            code,out,err=command(client,'tail -n 12 /workspace/boot.log',timeout=20)
            for secret in [os.environ.get('HF_TOKEN'),os.environ.get('RUNPOD_API_KEY')]:
                if secret:out=out.replace(secret,'[REDACTED]');err=err.replace(secret,'[REDACTED]')
            print(out[-5000:])
    elif sys.argv[1]=='finish':
        record=json.loads((OUT/'pod_created.json').read_text());safe_finish(record['id'],'operator_verified_finish')
