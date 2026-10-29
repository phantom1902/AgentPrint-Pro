"""Offline checks for the two collection changes made during import."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1] / 'data_collection'


def load(relative):
    spec = importlib.util.spec_from_file_location('collection_check', ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CollectionImportTests(unittest.TestCase):
    def test_session_lookup_matches_workspace_not_encoded_directory_name(self):
        module = load('scripts/copy_cc_mimo_sessions.py')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module.EPISODES_DIR = root / 'episodes'
            module.CLAUDE_PROJECTS = root / 'projects'
            unrelated = module.CLAUDE_PROJECTS / 'a-unrelated'
            matching = module.CLAUDE_PROJECTS / 'arbitrary-directory-name'
            unrelated.mkdir(parents=True)
            matching.mkdir()
            workspace = module.EPISODES_DIR / 'A1700' / 'workspace'
            (unrelated / 'session.jsonl').write_text(json.dumps({'cwd': str(root / 'other')}))
            (matching / 'session.jsonl').write_text(
                'malformed\n[]\n' + json.dumps({'cwd': str(workspace)}) + '\n')
            self.assertEqual(module.get_workspace_project_dir('A1700'), matching)
            self.assertIsNone(module.get_workspace_project_dir('A1701'))

    def test_missing_runtime_stops_before_ports_or_fixtures(self):
        module = load('tools/smoke_collection.py')
        with tempfile.TemporaryDirectory() as tmp:
            module.ROOT = Path(tmp)
            with patch('sys.argv', ['smoke_collection.py']), \
                    patch.object(module.socket, 'socket') as socket, \
                    patch.object(module.subprocess, 'run') as run, \
                    contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as exc:
                    module.main()
            self.assertEqual(exc.exception.code, 2)
            self.assertIn('runtime is incomplete', error.getvalue())
            socket.assert_not_called()
            run.assert_not_called()
            self.assertEqual(list(module.ROOT.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
