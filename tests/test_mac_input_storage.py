"""Mac storage optimization is performed on new, closed bundles only."""

import errno
import json
import os
import sys
from pathlib import Path

import pytest

from evidence_lens import _capacity, _dedup, archive, recorder
from evidence_lens._bundle import load, sha
from evidence_lens._verify import verify_bundle, verify_dependencies


@pytest.mark.skipif(sys.platform != 'darwin', reason='Mac prepublication hard-link path')
def test_worker_input_isolation_then_sealed_local_and_received_dedup(tmp_path, monkeypatch, capsys):
    store, source = tmp_path/'store', tmp_path/'project'
    (source/'scripts').mkdir(parents=True)
    (source/'scripts/stub.py').write_text('# fixture\n')
    config = tmp_path/'config.json'
    config.write_text('{}')
    raw = tmp_path/'raw.bin'
    raw.write_bytes(bytes(range(256))*4096)
    def stage(kind, inputs, parents, command, expected=0):
        args = ['--root',str(store),'--source-root',str(source),'--kind',kind,'--config',str(config),
                '--experiment','link-unit','--trial','test']
        for name, path in inputs.items():
            args += ['--input',f'{name}={path}']
        for rid in parents:
            args += ['--parent',rid]
        rc = recorder.main([*args,'--','{python}','-c',*command]); report = capsys.readouterr()
        assert rc == expected, (report.out, report.err)
        return json.loads(report.out.strip().splitlines()[-1])
    base = stage('scene_render',{'raw':raw},[],['import shutil,sys;shutil.copyfile(sys.argv[1],sys.argv[2]+"/source.bin")','{input:raw}','{out}'])
    parent = Path(base['path'])
    original = parent/'payload/source.bin'
    before = original.stat()
    assert before.st_mode & 0o222 == 0
    pin = sha(original)
    code = ('import json,os,pathlib,sys;p=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);'
            'assert p.stat().st_ino != int(sys.argv[3]);data=p.read_bytes();'
            'p.chmod(0o600);p.write_bytes(b"private worker copy");p.write_bytes(data);p.chmod(0o444);'
            '(out/"worker-inode.json").write_text(json.dumps(p.stat().st_ino));sys.exit(int(sys.argv[4]))')
    def measure(exit_code):
        record = stage('derivation',{'selected':original},[parent.name],
                       [code,'{input:selected}','{out}',str(before.st_ino),str(exit_code)],exit_code)
        bundle = Path(record['path']); private = bundle/'inputs/selected/source.bin'
        assert json.loads((bundle/'payload/worker-inode.json').read_text()) != before.st_ino
        assert sha(original) == sha(private) == pin
        current = original.stat()
        assert (current.st_ino,current.st_mode,current.st_mtime_ns) == (before.st_ino,before.st_mode,before.st_mtime_ns)
        assert verify_bundle(bundle)['run']['execution_status'] == ('succeeded' if exit_code == 0 else 'failed')
        return record, private
    child, linked = measure(0)
    assert linked.samefile(original) and child['local_input_storage']['linked_input_bytes'] == len(raw.read_bytes())
    failed, failed_input = measure(7)
    assert failed_input.samefile(original) and failed['exit_code'] == 7
    with monkeypatch.context() as mp:
        real_link = os.link
        def unsupported(src, dst, *args, **kwargs):
            if Path(dst).name.startswith('.parent-link-'):
                raise OSError(errno.EXDEV,'simulated unsupported link')
            return real_link(src,dst,*args,**kwargs)
        mp.setattr(_dedup.os,'link',unsupported)
        fallback, independent = measure(0)
    assert not independent.samefile(original) and fallback['local_input_storage']['linked_files'] == 0
    assert _capacity.bytes_on_tree(store, count_links=True) - _capacity.bytes_on_tree(store) == 2*raw.stat().st_size
    receiver = tmp_path/'receiver'
    assert archive.receive(parent,receiver)['status'] == 'accepted'
    pack = tmp_path/'child.zip'
    archive.pack(Path(child['path']),pack,store)
    incoming = archive.receive(pack,receiver,sha(pack))
    assert incoming['status'] == 'accepted' and incoming['local_input_storage']['linked_input_bytes'] == raw.stat().st_size
    peer = receiver/'runs'/child['run_id']
    imported = peer/'inputs/selected/source.bin'
    target = receiver/'runs'/parent.name/'payload/source.bin'
    assert imported.samefile(target) and not imported.samefile(original)
    assert archive.receive(pack,receiver,sha(pack))['status'] == 'identical'
    assert imported.samefile(target) and sha(target) == pin and target.stat().st_mode & 0o222 == 0
    verified = verify_bundle(peer)
    assert verified['manifest_sha256'] == child['manifest_sha256']
    verify_dependencies(verified,receiver)
