"""Fresh private-Hub checkpoint download with manifest and optimizer file checks."""
import json
import os
import time

from huggingface_hub import HfApi, snapshot_download
from tools.rl_cloud import credentials
from .core import OUT, CONFIG, sha, write


def run():
    credentials()
    record = json.loads((OUT/'pod/latest_checkpoint.json').read_text(encoding='utf-8'))
    prefix = f"{CONFIG['checkpoint_prefix']}/step-{record['step']:04d}"
    api = HfApi()
    assert api.model_info(CONFIG['output_repo'], revision=record['commit']).private
    started = time.time()
    root = OUT/'hf_checkpoint_backup'
    snapshot_download(CONFIG['output_repo'], revision=record['commit'], local_dir=root,
        allow_patterns=[prefix+'/*'])
    source = root/prefix
    manifest = json.loads((source/'hashes.json').read_text(encoding='utf-8'))
    assert manifest == record['files']
    assert 'resume.pt' in manifest
    assert all(sha(source/name) == digest for name, digest in manifest.items())
    # Same-volume hard links avoid a second multi-GB copy. These checkpoint
    # files are immutable; the watchdog skips them once size/mtime match.
    destination = OUT/f"pod/checkpoints/step-{record['step']:04d}"
    destination.mkdir(parents=True, exist_ok=True)
    for name in [*manifest, 'hashes.json']:
        target = destination/name
        if target.exists():
            assert sha(target) == sha(source/name)
        else:
            os.link(source/name, target)
    result = {'time': time.time(), 'repo': CONFIG['output_repo'], 'commit': record['commit'],
        'step': record['step'], 'files_verified': len(manifest), 'optimizer_rng_included': True,
        'fresh_download': True, 'path': str(destination), 'seconds': time.time()-started}
    write('checkpoint_local_backup.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    run()
