"""Offline preflight checks; no GPU, Warden or media needed."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from src import preflight


class PreflightTests(unittest.TestCase):
    def test_warden_unreachable_busy_and_unregistered_fail(self):
        self.assertEqual([r.ok for r in preflight.check_warden(None)], [False])
        busy = preflight.check_warden({'state': 'ready', 'refcount': 2, 'loaded_model': preflight.QWEN_MODEL,
                                       'registry': [preflight.QWEN_MODEL]})
        self.assertEqual([r.ok for r in busy], [True, False])
        missing = preflight.check_warden({'state': 'idle', 'refcount': 0, 'registry': ['other']})
        self.assertEqual([r.ok for r in missing], [False, True])

    def test_gpu_memory_outside_the_warden_tree_is_foreign(self):
        parents = {101: 50, 50: 1, 202: 1}
        apps = '101, llama-server, 20000\n202, python, 3000\n303, python, 200\n404, other, [N/A]\n'
        with patch.object(preflight, '_parent', side_effect=lambda pid: parents.get(pid)):
            self.assertEqual(preflight.foreign_gpu(apps, warden=50), ['pid 202 python 3000 MiB'])
            self.assertEqual(len(preflight.foreign_gpu(apps, warden=None)), 2)

    def test_missing_media_fails(self):
        [result] = preflight.check_media([Path('/nonexistent/episode.mkv')])
        self.assertFalse(result.ok)

    def test_report_prints_fix_and_exit_status(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(preflight.report([preflight.Result(True, 'a', 'fine')]), 0)
            self.assertEqual(preflight.report([preflight.Result(False, 'b', 'broken', 'do this')]), 1)
        self.assertIn('fix: do this', stream.getvalue())
        self.assertIn('preflight: ready', stream.getvalue())


if __name__ == '__main__':
    unittest.main()
