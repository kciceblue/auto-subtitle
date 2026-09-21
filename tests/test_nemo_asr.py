"""Offline invariants for optional isolated NeMo recognition; no model loading."""
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from src import nemo_asr as n
from src.workflow_state import file_hash, read_json, write_json


@dataclass
class Hypothesis:
    text: str
    score: object = None
    y_sequence: object = None
    timestamp: object = None


def response(spec):
    return {"request_key": spec["request_key"], "identity": spec["identity"],
            "complete": True, "error": None,
            "windows": [dict(row, text=f" raw {i}。 ", decode_seconds=.1,
                             raw_return=n._raw_json([Hypothesis(f" raw {i}。 ")]))
                        for i, row in enumerate(spec["windows"])]}


class NemoBridgeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.model = self.directory / "official.nemo"; self.model.write_bytes(b"weights")
        self.python = self.directory / "env/bin/python"; self.python.parent.mkdir(parents=True)
        self.python.symlink_to(Path(os.sys.executable))
        (self.python.parent.parent / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
        self.config = dict(nemo_model=self.model, nemo_python=self.python,
                           warden_admin_url="http://local.test/admin", unload_warden_before_asr=False)
        self.audio = np.arange(160,dtype=np.float32)/1000
        self.windows = [SimpleNamespace(index=7,start=.001,end=.003),
                        SimpleNamespace(index=9,start=.004,end=.008)]
        self.identity = {"model":str(self.model), "python":str(self.python), "runtime": {"version":"test"}}

    def test_original_samples_and_input_order_survive_private_packet(self):
        original = self.audio.copy(); seen = []
        def launch(spec,directory):
            seen.append((spec, directory))
            self.assertEqual(spec["warden_admin_url"], self.config["warden_admin_url"])
            self.assertFalse(spec["unload_warden_before_asr"])
            self.assertNotIn("context", spec)
            with np.load(spec["clips"], allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["clip_000000"], original[16:48])
                np.testing.assert_array_equal(arrays["clip_000001"], original[64:128])
            return response(spec)
        with patch.object(n,"runtime_identity",return_value=self.identity),patch.object(n,"_launch_worker",side_effect=launch):
            text, report = n.transcribe_windows(self.audio,16000,self.windows,self.config)
        self.assertEqual(text,[" raw 0。 "," raw 1。 "])
        self.assertEqual([r["window"] for r in report["windows"]],[7,9])
        np.testing.assert_array_equal(self.audio,original)
        self.assertFalse(seen[0][1].exists())
        json.dumps(report,ensure_ascii=False)

    def test_invalid_audio_or_windows_never_launch(self):
        variants=[(self.audio.astype(np.float64),16000,self.windows),
                  (self.audio,8000,self.windows),
                  (self.audio,16000,[self.windows[0],self.windows[0]]),
                  (self.audio,16000,[dict(index=1,start=0,end=1)]),
                  (self.audio,16000,[dict(index=1,start=0,end=float("nan"))])]
        with patch.object(n,"_launch_worker") as launch:
            for audio,sr,windows in variants:
                with self.subTest(sr=sr,windows=windows), self.assertRaises(ValueError):
                    n.transcribe_windows(audio,sr,windows,self.config)
            launch.assert_not_called()

    def test_wrong_hash_order_missing_rows_or_changed_text_cannot_pass(self):
        for flaw in ("hash","order","missing","text","complete","identity"):
            def launch(spec,directory):
                result=response(spec)
                if flaw=="hash":result["windows"][0]["float32_sha256"]="stale"
                if flaw=="order":result["windows"].reverse()
                if flaw=="missing":result["windows"].pop()
                if flaw=="text":result["windows"][0]["text"]="unsupported"
                if flaw=="complete":result["complete"]=False
                if flaw=="identity":result["identity"]={}
                return result
            with self.subTest(flaw=flaw),patch.object(n,"runtime_identity",return_value=self.identity),patch.object(n,"_launch_worker",side_effect=launch):
                with self.assertRaises(n.NemoASRError) as caught:
                    n.transcribe_windows(self.audio,16000,self.windows,self.config)
                self.assertIn("windows",caught.exception.report)
                self.assertFalse(caught.exception.report["complete"])

    def test_frozen_runtime_mismatch_rejects_before_launch(self):
        with patch.object(n,"runtime_identity",return_value=self.identity),patch.object(n,"_launch_worker") as launch:
            with self.assertRaisesRegex(n.NemoASRError,"cache"):
                n.transcribe_windows(self.audio,16000,self.windows,self.config,expected_identity={"old":True})
            launch.assert_not_called()

    def test_failed_subprocess_retains_raw_partial_report(self):
        def run(command,**kwargs):
            result={"complete":False,"error":"CUDA failed","windows":[{"raw_return":["untouched"]}]}
            write_json(Path(command[-1]),result)
            return SimpleNamespace(returncode=1,stdout="progress",stderr="failure")
        spec={"identity":self.identity}
        with patch.object(n.subprocess,"run",side_effect=run),self.assertRaises(n.NemoASRError) as caught:
            n._launch_worker(spec,self.directory)
        self.assertEqual(caught.exception.report["windows"][0]["raw_return"],["untouched"])
        self.assertEqual(caught.exception.report["exit_code"],1)
        self.assertTrue(caught.exception.report["worker_exited"])

    def test_timeout_has_no_partial_success_and_keeps_available_evidence(self):
        write_json(self.directory/"response.json",{"complete":False,"windows":[{"raw_return":"evidence"}]})
        with patch.object(n.subprocess,"run",side_effect=subprocess.TimeoutExpired(["worker"],1800)):
            with self.assertRaises(n.NemoASRError) as caught:n._launch_worker({"identity":self.identity},self.directory)
        self.assertFalse(caught.exception.report["complete"])
        self.assertEqual(caught.exception.report["windows"],[{"raw_return":"evidence"}])

    def test_runtime_identity_preserves_venv_launcher_and_binds_metadata(self):
        runtime={"python_version":"test", "packages":{key:{"version":"1","metadata_sha256":"sha"} for key in n.RUNTIME_PACKAGES}}
        result=SimpleNamespace(returncode=0,stdout=json.dumps(runtime),stderr="")
        with patch.object(n.subprocess,"run",return_value=result) as run:
            identity=n.runtime_identity(self.config)
        self.assertEqual(run.call_args.args[0][0],str(self.python))
        self.assertEqual(identity["python"],str(self.python))
        self.assertNotEqual(identity["python_resolved"],str(self.python))
        self.assertEqual(identity["model_sha256"],file_hash(self.model))
        self.assertEqual(identity["runtime"],runtime)
        self.assertIsNotNone(identity["pyvenv_config_sha256"])

    def test_worker_preserves_decoder_and_admission_contract_with_raw_evidence(self):
        self._mock_worker_case(malformed_second=False)

    def test_worker_preserves_malformed_raw_return_before_failing_whole_result(self):
        self._mock_worker_case(malformed_second=True)

    def _mock_worker_case(self, *, malformed_second):
        arrays,jobs=n._clips(self.audio,16000,self.windows)
        packet=self.directory/"clips.npz"; np.savez(packet,**arrays)
        runtime={"packages":{}, "python_version":"test"}
        identity=dict(self.identity,runtime=runtime,model_sha256=file_hash(self.model),
                      bridge_sha256=file_hash(Path(n.__file__)))
        spec=dict(identity=identity,request_key="prepared",windows=jobs,clips=str(packet),
                  clips_sha256=file_hash(packet),sample_rate=16000,
                  warden_admin_url="http://local.test/admin",unload_warden_before_asr=False)
        request=self.directory/"request.json"; output=self.directory/"response.json"
        write_json(request,spec)
        cfg={"decoding":{"strategy":"alsd","beam":{"beam_size":4,"return_best_hypothesis":True}}}
        model=Mock(); model.to.return_value=model; model.eval.return_value=model
        model.cfg=cfg; model.parameters.side_effect=lambda:iter([torch.zeros(1)])
        model.transcribe.side_effect=[[Hypothesis(" first ")],
                                     [Hypothesis("one"),Hypothesis("two")] if malformed_second else [Hypothesis(" second ")]]
        restore=Mock(return_value=model)
        modules={"nemo.collections.asr.models":SimpleNamespace(ASRModel=SimpleNamespace(restore_from=restore)),
                 "omegaconf":SimpleNamespace(OmegaConf=SimpleNamespace(to_container=lambda value,resolve:value))}
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.sys.modules,modules))
            stack.enter_context(patch.object(n,"_distribution_identity",return_value=runtime))
            admission=stack.enter_context(patch("src.warden.ensure_gpu_headroom",return_value=12e9))
            for name,value in (("is_available",True),("is_initialized",False),("synchronize",None),("reset_peak_memory_stats",None)):
                stack.enter_context(patch.object(torch.cuda,name,return_value=value))
            stack.callback(setattr,torch.backends.cuda.matmul,"allow_tf32",torch.backends.cuda.matmul.allow_tf32)
            stack.callback(setattr,torch.backends.cudnn,"allow_tf32",torch.backends.cudnn.allow_tf32)
            stack.enter_context(patch.object(torch,"set_float32_matmul_precision"))
            stack.enter_context(patch.object(torch,"manual_seed"))
            stack.enter_context(patch.object(torch,"set_num_threads"))
            if malformed_second:
                with self.assertRaisesRegex(RuntimeError,"exactly one"):n._worker(request,output)
            else:n._worker(request,output)
            admission.assert_called_once_with(required_gb=8.,hard_floor_gb=6.,
                                               admin_url="http://local.test/admin",enabled=False,caller="NeMo ASR")
        report=read_json(output)
        self.assertEqual(report["complete"],not malformed_second)
        self.assertEqual(report["restored_model_config"],cfg)
        self.assertEqual(report["windows"][0]["text"]," first ")
        self.assertEqual(len(report["windows"]),2)
        self.assertEqual(model.transcribe.call_count,2)
        for call in model.transcribe.call_args_list:
            self.assertEqual(call.kwargs,dict(batch_size=1,return_hypotheses=True,num_workers=0,verbose=False,timestamps=None))
        np.testing.assert_array_equal(model.transcribe.call_args_list[0].args[0][0],self.audio[16:48])
        if malformed_second:
            self.assertEqual(len(report["windows"][1]["raw_return"]),2)
            self.assertNotIn("text",report["windows"][1])

    def test_raw_serialization_preserves_text_tokens_timing_and_nonfinite_score(self):
        raw=Hypothesis(" exact 。 ",float("-inf"),torch.tensor([4,8]),np.array([1.25,2.5]))
        value=n._raw_json([raw])[0]["fields"]
        self.assertEqual(value["text"],raw.text)
        self.assertEqual(value["score"],{"nonfinite_float":"-inf"})
        self.assertEqual(value["y_sequence"]["values"],[4,8])
        self.assertEqual(value["timestamp"]["values"],[1.25,2.5])
        self.assertEqual(n._one_text([raw]),raw.text)
        for result in ([],[raw,raw],([raw],[raw]),["text"]):
            with self.assertRaises(ValueError):n._one_text(result)
        with self.assertRaises(TypeError):n._raw_json(object())


if __name__ == "__main__":
    unittest.main()
