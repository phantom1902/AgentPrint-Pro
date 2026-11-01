"""V2 public evaluation/recovery must validate the actual base cache."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import unittest

import numpy as np
import yaml

from agentprint.__main__ import main
from agentprint.config import fingerprint
import test_agentprint_v2 as tracer


class V2BaseIntegrity(unittest.TestCase):
    setUp = tracer.V2Commands.setUp

    def command(self, command, *args, ok=True):
        self.config.write_text(yaml.safe_dump(self.settings), encoding='utf-8')
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            status = main([command, '--config', str(self.config), *args])
        self.assertEqual(status == 0, ok, out.getvalue() + err.getvalue())
        return out.getvalue(), err.getvalue()

    def test_disabled_iat_evaluation_and_resume_reject_actual_cache_identity_changes(self):
        self.command('prepare')
        out, _ = self.command('train')
        run = Path(json.loads(out.splitlines()[-1])['run_dir'])
        self.command('evaluate', '--checkpoint', str(run / 'best.pt'))
        self.command('train', '--resume', str(run))
        base = self.root / 'cache/scenario/val'
        manifest, metadata = base / 'valid.jsonl', base / 'metadata.json'
        original_manifest, original_metadata = manifest.read_bytes(), metadata.read_bytes()
        row = json.loads(original_manifest)
        shard = self.root / 'cache' / row['shard_relpath']
        original_shard = shard.read_bytes()

        def reject(message):
            for command, args in [('evaluate', ('--checkpoint', str(run / 'best.pt'))),
                                  ('train', ('--resume', str(run)))]:
                with self.subTest(command=command, corruption=message):
                    _, error = self.command(command, *args, ok=False)
                    self.assertIn(message, error)

        changed = dict(row, label_id=4)
        manifest.write_text(json.dumps(changed) + '\n', encoding='utf-8')
        reject('manifest identity')
        manifest.write_bytes(original_manifest)
        for field, value, message in [('valid_count', 99, 'valid count'),
                ('preprocessing_version', 'foreign-version', 'preprocessing_version')]:
            values = json.loads(original_metadata)
            values[field] = value
            metadata.write_text(json.dumps(values), encoding='utf-8')
            reject(message)
            metadata.write_bytes(original_metadata)
        for values in (np.zeros((1, 2, 2, 1800), dtype=np.float32),
                       np.zeros((1, 2, 2, 900), dtype=np.uint64)):
            np.save(shard, values, allow_pickle=False)
            reject('shard array')
            shard.write_bytes(original_shard)
        changed = dict(row, row_index=1)
        manifest.write_text(json.dumps(changed) + '\n', encoding='utf-8')
        values = json.loads(original_metadata)
        values['manifest_id'] = fingerprint([changed])
        metadata.write_text(json.dumps(values), encoding='utf-8')
        reject('shard array')
        manifest.write_bytes(original_manifest)
        metadata.write_bytes(original_metadata)
        self.command('evaluate', '--checkpoint', str(run / 'best.pt'))
        self.command('train', '--resume', str(run))


if __name__ == '__main__':
    unittest.main()
