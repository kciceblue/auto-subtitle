"""Mocked local writer lifecycle checks; no server, GPU, or network calls."""
from contextlib import ExitStack
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from src import local_backend as backend


class LocalBackendTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.model = self.directory / "writer.gguf"
        self.model.write_bytes(b"test weights")
        self.binary = self.directory / "llama-server"
        self.binary.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        self.binary.chmod(0o700)
        self.options = {"alias": "local-writer", "log_path": self.directory / "logs/server.log",
                        "expected_sha256": hashlib.sha256(b"test weights").hexdigest()}
        self.props = {"model_path": str(self.model), "model_alias": "local-writer",
                      "chat_template": "template", "build_info": "mock-build"}
        self.process = Mock(pid=123, returncode=None)
        self.process.poll.return_value = None
        self.reserved = Mock()
        self.http_calls = []

    def response(self, url, payload=None, **kwargs):
        self.http_calls.append((url, payload, kwargs))
        if url.endswith("/props"):
            return self.props
        if url.endswith("/load"):
            return {"loaded": "qwen3.8-27b-dflash"}
        if url.endswith("/unload"):
            return {"unloaded": True}
        raise AssertionError("Unexpected service call")

    def patches(self, *, response=None, popen=None, reserve=None):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.http = stack.enter_context(patch.object(backend, "_request_json", side_effect=response or self.response))
        self.launch = stack.enter_context(patch.object(backend.subprocess, "Popen", **(
            {"side_effect": popen} if popen else {"return_value": self.process})))
        self.reserve = stack.enter_context(patch.object(backend, "_reserve_port", **(
            {"side_effect": reserve} if reserve else {"return_value": self.reserved})))
        return stack

    def writer(self, **changes):
        return backend.temporary_local_writer(self.model, self.binary, **dict(self.options, **changes))

    def projector_fixture(self):
        projector = self.directory / "audio-projector.gguf"
        projector.write_bytes(b"projector weights")
        return projector, hashlib.sha256(projector.read_bytes()).hexdigest()

    def test_projector_pair_and_valid_pin_are_required_before_unload(self):
        projector, pin = self.projector_fixture()
        self.patches()
        changes = [{"projector_path": projector}, {"projector_sha256": pin},
                   {"projector_path": projector, "projector_sha256": "0" * 64},
                   {"projector_path": projector, "projector_sha256": 123},
                   {"projector_path": projector, "projector_sha256": "not-a-hash"}]
        for options in changes:
            with self.subTest(options=options), self.assertRaises(ValueError):
                with self.writer(**options):
                    self.fail("Invalid projector accepted")
        self.http.assert_not_called()
        self.reserve.assert_not_called()
        self.launch.assert_not_called()

    def test_projector_must_be_an_existing_gguf_file_before_unload(self):
        projector, pin = self.projector_fixture()
        invalid = self.directory / "projector.txt"
        invalid.write_bytes(projector.read_bytes())
        self.patches()
        for path in [invalid, self.directory, self.directory / "missing.gguf"]:
            with self.subTest(path=path), self.assertRaises((ValueError, OSError)):
                with self.writer(projector_path=path, projector_sha256=pin):
                    self.fail("Invalid projector path accepted")
        self.http.assert_not_called()
        self.launch.assert_not_called()

    def test_projector_is_passed_and_recorded_with_split_text_weights(self):
        _, shards = self.split_fixture()
        projector, pin = self.projector_fixture()
        self.patches()
        with self.writer(expected_shards=shards, projector_path=projector,
                         projector_sha256=pin.upper()) as (_, identity):
            self.assertEqual(identity["projector_path"], str(projector))
            self.assertEqual(identity["projector_sha256"], pin)
            self.assertEqual(identity["projector_file"],
                             {"path": str(projector), "sha256": pin,
                              "bytes": projector.stat().st_size})
            self.assertEqual(len(identity["model_files"]), 2)
            command = self.launch.call_args.args[0]
            self.assertEqual(command[command.index("--mmproj") + 1], str(projector))
            self.assertNotIn("--media-path", command)
        self.process.terminate.assert_called_once()
        self.assertTrue(self.http_calls[-1][0].endswith("/load"))

    def test_projector_mutation_while_hashing_never_contacts_model_service(self):
        projector, pin = self.projector_fixture()
        self.patches()
        original = backend._sha256
        def digest(path):
            answer = original(path)
            if path == projector:
                path.write_bytes(b"changed during projector hashing")
            return answer
        with patch.object(backend, "_sha256", side_effect=digest):
            with self.assertRaisesRegex(backend.LocalBackendError, "during verification"):
                with self.writer(projector_path=projector, projector_sha256=pin):
                    self.fail("Changed projector accepted")
        self.http.assert_not_called()
        self.launch.assert_not_called()

    def test_projector_mutation_during_startup_cleans_up_and_restores(self):
        projector, pin = self.projector_fixture()
        def response(url, *args, **kwargs):
            result = self.response(url, *args, **kwargs)
            if url.endswith("/props"):
                projector.write_bytes(b"changed projector during startup")
            return result
        self.patches(response=response)
        with self.assertRaisesRegex(backend.LocalBackendError, "during model startup"):
            with self.writer(projector_path=projector, projector_sha256=pin):
                self.fail("Changed projector accepted")
        self.process.terminate.assert_called_once()
        self.assertTrue(self.http_calls[-1][0].endswith("/load"))

    def test_projector_mutation_during_inference_rejects_result_and_restores(self):
        projector, pin = self.projector_fixture()
        self.patches()
        with self.assertRaisesRegex(backend.LocalBackendError, "during inference"):
            with self.writer(projector_path=projector, projector_sha256=pin):
                previous = projector.stat()
                projector.write_bytes(b"PROJECTOR WEIGHTS")
                os.utime(projector, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        self.process.terminate.assert_called_once()
        self.assertTrue(self.http_calls[-1][0].endswith("/load"))

    def test_log_cannot_overwrite_projector_even_through_file_aliases(self):
        projector, pin = self.projector_fixture()
        link = self.directory / "projector-symlink.log"
        link.symlink_to(projector)
        hardlink = self.directory / "projector-hardlink.log"
        hardlink.hardlink_to(projector)
        self.patches()
        for path in [projector, link, hardlink]:
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "log must not overwrite"):
                with self.writer(projector_path=projector, projector_sha256=pin, log_path=path):
                    self.fail("Projector could be modified by logging")
        self.assertEqual(projector.read_bytes(), b"projector weights")
        self.http.assert_not_called()
        self.launch.assert_not_called()

    def test_existing_separate_log_remains_append_only(self):
        projector, pin = self.projector_fixture()
        log = self.directory / "existing.log"
        log.write_text("earlier run\n", encoding="utf-8")
        self.patches()
        with self.writer(projector_path=projector, projector_sha256=pin, log_path=log):
            pass
        self.assertEqual(log.read_text(encoding="utf-8"), "earlier run\n")

    def template_fixture(self):
        template = self.directory / "audio-template.jinja"
        template.write_text(self.props["chat_template"], encoding="utf-8")
        return template, hashlib.sha256(template.read_bytes()).hexdigest()

    def test_template_pair_and_correct_pin_are_required_before_unload(self):
        template, pin = self.template_fixture()
        self.patches()
        for options in ({"chat_template_path": template}, {"chat_template_sha256": pin},
                        {"chat_template_path": template, "chat_template_sha256": "0" * 64},
                        {"chat_template_path": template, "chat_template_sha256": 123},
                        {"chat_template_path": self.directory, "chat_template_sha256": pin}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                with self.writer(**options):
                    self.fail("Invalid chat template accepted")
        self.http.assert_not_called()
        self.launch.assert_not_called()

    def test_template_override_records_actual_hash_and_works_with_projector(self):
        template, template_pin = self.template_fixture()
        projector, projector_pin = self.projector_fixture()
        self.patches()
        with self.writer(chat_template_path=template, chat_template_sha256=template_pin.upper(),
                         projector_path=projector, projector_sha256=projector_pin) as (_, identity):
            command = self.launch.call_args.args[0]
            self.assertEqual(command[command.index("--chat-template-file") + 1], str(template))
            self.assertEqual(command[command.index("--mmproj") + 1], str(projector))
            self.assertEqual(identity["chat_template_path"], str(template))
            self.assertEqual(identity["chat_template_sha256"], template_pin)
            self.assertEqual(identity["chat_template_file_sha256"], template_pin)
            self.assertEqual(identity["chat_template_file"],
                             {"path": str(template), "sha256": template_pin,
                              "bytes": template.stat().st_size})
        self.process.terminate.assert_called_once()

    def test_different_reported_template_rejects_and_restores(self):
        template, pin = self.template_fixture()
        self.props["chat_template"] = "another template"
        self.patches()
        with self.assertRaisesRegex(backend.LocalBackendError, "different chat template"):
            with self.writer(chat_template_path=template, chat_template_sha256=pin):
                self.fail("Unbound served template accepted")
        self.process.terminate.assert_called_once()
        self.assertTrue(self.http_calls[-1][0].endswith("/load"))

    def test_template_mutation_while_hashing_never_unloads(self):
        template, pin = self.template_fixture()
        self.patches()
        original = backend._sha256
        def digest(path):
            answer = original(path)
            if path == template:
                path.write_text("changed template", encoding="utf-8")
            return answer
        with patch.object(backend, "_sha256", side_effect=digest):
            with self.assertRaisesRegex(backend.LocalBackendError, "during verification"):
                with self.writer(chat_template_path=template, chat_template_sha256=pin):
                    self.fail("Changed template accepted")
        self.http.assert_not_called()

    def test_template_mutation_during_startup_rejects_and_restores(self):
        template, pin = self.template_fixture()
        def response(url, *args, **kwargs):
            result = self.response(url, *args, **kwargs)
            if url.endswith("/props"):
                template.write_text("changed template", encoding="utf-8")
            return result
        self.patches(response=response)
        with self.assertRaisesRegex(backend.LocalBackendError, "during model startup"):
            with self.writer(chat_template_path=template, chat_template_sha256=pin):
                self.fail("Changed template accepted")
        self.process.terminate.assert_called_once()
        self.assertTrue(self.http_calls[-1][0].endswith("/load"))

    def test_template_mutation_during_inference_rejects_and_restores(self):
        template, pin = self.template_fixture()
        self.patches()
        with self.assertRaisesRegex(backend.LocalBackendError, "during inference"):
            with self.writer(chat_template_path=template, chat_template_sha256=pin):
                previous = template.stat()
                template.write_text("TEMPLATE", encoding="utf-8")
                os.utime(template, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        self.process.terminate.assert_called_once()
        self.assertTrue(self.http_calls[-1][0].endswith("/load"))

    def test_log_cannot_alias_the_pinned_template(self):
        template, pin = self.template_fixture()
        symlink = self.directory / "template-symlink.log"
        symlink.symlink_to(template)
        hardlink = self.directory / "template-hardlink.log"
        hardlink.hardlink_to(template)
        self.patches()
        for log in (template, symlink, hardlink):
            with self.subTest(log=log), self.assertRaisesRegex(ValueError, "log must not overwrite"):
                with self.writer(chat_template_path=template, chat_template_sha256=pin, log_path=log):
                    self.fail("Template could be modified by logging")
        self.assertEqual(template.read_text(encoding="utf-8"), "template")
        self.http.assert_not_called()

    def test_unchanged_weight_checks_reuse_only_process_local_verification(self):
        with patch.object(backend, '_sha256', wraps=backend._sha256) as digest:
            backend._weight_bundle(self.model, self.options['expected_sha256'], None)
            backend._weight_bundle(self.model, self.options['expected_sha256'], None)
            self.assertEqual(digest.call_count, 1)

    def test_same_size_edit_with_restored_mtime_cannot_reuse_verification(self):
        backend._weight_bundle(self.model, self.options['expected_sha256'], None)
        stat = self.model.stat()
        self.model.write_bytes(b'TEST WEIGHTS')
        self.assertEqual(self.model.stat().st_size, stat.st_size)
        os.utime(self.model, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            backend._weight_bundle(self.model, self.options['expected_sha256'], None)

    def test_file_mutation_during_hash_is_rejected_before_model_service(self):
        self.patches()
        def digest(path):
            answer = hashlib.sha256(path.read_bytes()).hexdigest()
            path.write_bytes(b'changed while hashing')
            return answer
        with patch.object(backend, '_sha256', side_effect=digest):
            with self.assertRaisesRegex(backend.LocalBackendError, 'during verification'):
                with self.writer():
                    self.fail('Changed weights accepted')
        self.http.assert_not_called()

    def test_weight_mutation_during_inference_rejects_result_and_restores(self):
        self.patches()
        with self.assertRaisesRegex(backend.LocalBackendError, 'during inference'):
            with self.writer():
                self.model.write_bytes(b'changed during inference')
        self.process.terminate.assert_called_once()
        self.assertTrue(self.http_calls[-1][0].endswith('/load'))

    def split_fixture(self):
        self.model = self.model.with_name('writer-00001-of-00002.gguf')
        self.model.write_bytes(b'first shard')
        second = self.model.with_name('writer-00002-of-00002.gguf')
        second.write_bytes(b'second shard')
        pins = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.model, second)}
        self.options['expected_sha256'] = pins[self.model.name]
        self.props['model_path'] = str(self.model)
        return second, pins

    def test_split_weights_require_every_pin_before_unload(self):
        _, pins = self.split_fixture()
        self.patches()
        with self.assertRaisesRegex(ValueError, 'Pin every'):
            with self.writer(expected_shards={self.model.name: pins[self.model.name]}):
                self.fail('Missing pin accepted')
        self.http.assert_not_called()
        self.launch.assert_not_called()

    def test_split_identity_and_cpu_expert_offload_are_recorded(self):
        _, pins = self.split_fixture()
        self.patches()
        with self.writer(expected_shards=pins, cpu_moe_layers=36, threads=16) as (_, identity):
            self.assertEqual(len(identity['model_files']), 2)
            self.assertEqual(identity['cpu_moe_layers'], 36)
            self.assertEqual(identity['threads'], 16)
            command = self.launch.call_args.args[0]
            self.assertEqual(command[command.index('--n-cpu-moe')+1], '36')
            self.assertEqual(command[command.index('--threads')+1], '16')

    def test_secondary_shard_mutation_during_startup_is_rejected(self):
        second, pins = self.split_fixture()
        def response(url, *args, **kwargs):
            result = self.response(url, *args, **kwargs)
            if url.endswith('/props'):
                second.write_bytes(b'changed second shard')
            return result
        self.patches(response=response)
        with self.assertRaisesRegex(backend.LocalBackendError, 'weights changed'):
            with self.writer(expected_shards=pins):
                self.fail('Changed model accepted')
        self.process.terminate.assert_called_once()
        self.assertTrue(any(call.args[0].endswith('/load') for call in self.http.call_args_list))

    def test_full_swa_is_explicit_in_command_and_identity(self):
        self.patches()
        with self.writer(full_swa=True) as (_, identity):
            self.assertTrue(identity['full_swa'])
            self.assertIn('--swa-full', self.launch.call_args.args[0])

    def test_success_yields_verified_identity_and_restores_after_owned_child_exits(self):
        events = []
        def response(*args, **kwargs):
            events.append(args[0].rsplit("/", 1)[-1])
            return self.response(*args, **kwargs)
        self.patches(response=response)
        self.process.terminate.side_effect = lambda: events.append("terminate")
        self.process.wait.side_effect = lambda **kwargs: events.append("wait")
        with self.writer() as (endpoint, identity):
            self.assertEqual(endpoint, "http://127.0.0.1:18101/v1/chat/completions")
            self.assertEqual(identity["model_path"], str(self.model))
            self.assertEqual(identity["model_sha256"], self.options["expected_sha256"])
            self.assertEqual(identity["model_alias"], "local-writer")
            command = self.launch.call_args.args[0]
            for flag, value in [("--parallel", "1"), ("--n-gpu-layers", "99"),
                                ("--cache-type-k", "q8_0"), ("--flash-attn", "on")]:
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertIn("--jinja", command)
            self.assertNotIn("--mmproj", command)
            self.assertNotIn("--chat-template-file", command)
            self.assertNotIn("chat_template_path", identity)
        self.assertEqual(events, ["unload", "props", "terminate", "wait", "load"])
        self.process.kill.assert_not_called()

    def test_occupied_port_prevents_unload_and_launch(self):
        self.patches(reserve=backend.LocalBackendError("occupied"))
        with self.assertRaisesRegex(backend.LocalBackendError, "occupied"):
            with self.writer():
                self.fail("must not yield")
        self.http.assert_not_called()
        self.launch.assert_not_called()

    def test_reservation_bind_failure_closes_socket(self):
        sock = Mock()
        sock.bind.side_effect = OSError("occupied")
        with patch.object(backend.socket, "socket", return_value=sock):
            with self.assertRaisesRegex(backend.LocalBackendError, "occupied"):
                backend._reserve_port()
        sock.close.assert_called_once()

    def test_invalid_hash_alias_or_remote_admin_never_contacts_service(self):
        self.patches()
        for changes in ({"expected_sha256": "0" * 64}, {"alias": "--unsafe"},
                        {"admin_url": "https://example.org/admin"}, {"context_size": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                with self.writer(**changes):
                    self.fail("must not yield")
        self.http.assert_not_called()
        self.reserve.assert_not_called()

    def test_spawn_failure_still_restores_and_preserves_original_exception(self):
        original = OSError("cannot launch")
        self.patches(popen=original)
        with self.assertRaises(OSError) as caught:
            with self.writer():
                self.fail("must not yield")
        self.assertIs(caught.exception, original)
        self.assertEqual([call[0].rsplit("/", 1)[-1] for call in self.http_calls], ["unload", "load"])

    def test_wrong_model_identity_stops_owned_child_and_restores(self):
        self.props["model_path"] = str(self.directory / "wrong.gguf")
        self.patches()
        with self.assertRaisesRegex(backend.LocalBackendError, "different model path"):
            with self.writer():
                self.fail("must not yield")
        self.process.terminate.assert_called_once()
        self.assertTrue(self.http_calls[-1][0].endswith("/load"))

    def test_wrong_reported_alias_is_rejected(self):
        self.props["model_alias"] = "other-writer"
        self.patches()
        with self.assertRaisesRegex(backend.LocalBackendError, "different model alias"):
            with self.writer():
                self.fail("must not yield")

    def test_body_exception_survives_restore_failure_with_note(self):
        original = ValueError("translation failed")
        def response(url, *args, **kwargs):
            if url.endswith("/load"):
                raise OSError("restore unavailable")
            return self.response(url, *args, **kwargs)
        self.patches(response=response)
        with self.assertLogs(backend.logger, level="ERROR"), self.assertRaises(ValueError) as caught:
            with self.writer():
                raise original
        self.assertIs(caught.exception, original)
        self.assertIn("restore unavailable", " ".join(caught.exception.__notes__))
        self.process.terminate.assert_called_once()

    def test_restore_failure_after_success_is_an_explicit_error(self):
        def response(url, *args, **kwargs):
            if url.endswith("/load"):
                return {"loaded": "wrong-model"}
            return self.response(url, *args, **kwargs)
        self.patches(response=response)
        with self.assertLogs(backend.logger, level="ERROR"), self.assertRaisesRegex(backend.LocalBackendError, "Restore model"):
            with self.writer():
                pass

    def test_readiness_timeout_cleans_up_and_restores(self):
        def response(url, *args, **kwargs):
            if url.endswith("/props"):
                raise OSError("still loading")
            return self.response(url, *args, **kwargs)
        self.patches(response=response)
        with patch.object(backend, "READY_TIMEOUT", 0), self.assertRaisesRegex(backend.LocalBackendError, "timed out"):
            with self.writer():
                self.fail("must not yield")
        self.process.terminate.assert_called_once()
        self.assertTrue(self.http_calls[-1][0].endswith("/load"))

    def test_terminate_timeout_kills_only_owned_process_before_restore(self):
        self.patches()
        self.process.wait.side_effect = [subprocess.TimeoutExpired(["owned"], 20), 0]
        with self.writer():
            pass
        self.process.terminate.assert_called_once()
        self.process.kill.assert_called_once()
        self.assertEqual(self.process.wait.call_count, 2)
        self.assertTrue(self.http_calls[-1][0].endswith("/load"))

    def test_unconfirmed_child_shutdown_does_not_load_competing_model(self):
        self.patches()
        self.process.terminate.side_effect = OSError("cannot terminate")
        with self.assertLogs(backend.logger, level="ERROR"), self.assertRaisesRegex(backend.LocalBackendError, "Restore skipped"):
            with self.writer():
                pass
        self.assertFalse(any(call[0].endswith("/load") for call in self.http_calls))

    def test_child_exit_race_during_termination_still_allows_restore(self):
        self.patches()
        def terminate():
            self.process.poll.return_value = 0
            raise ProcessLookupError("already exited")
        self.process.terminate.side_effect = terminate
        with self.writer():
            pass
        self.assertTrue(self.http_calls[-1][0].endswith("/load"))

    def test_early_child_exit_is_not_treated_as_ready(self):
        self.patches()
        self.process.poll.return_value = 7
        self.process.returncode = 7
        with self.assertRaisesRegex(backend.LocalBackendError, "startup: 7"):
            with self.writer():
                self.fail("must not yield")
        self.process.terminate.assert_not_called()
        self.assertTrue(self.http_calls[-1][0].endswith("/load"))

    def test_log_creation_failure_does_not_unload(self):
        self.patches()
        with self.assertRaises(OSError):
            with self.writer(log_path=self.model / "invalid/server.log"):
                self.fail("must not yield")
        self.http.assert_not_called()
        self.reserved.close.assert_called()


if __name__ == "__main__":
    unittest.main()
