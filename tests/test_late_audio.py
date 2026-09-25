"""Synthetic CPU-only late acoustic acquisition checks; no native model calls."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from src import late_audio as la


class GeometryTests(unittest.TestCase):
    def test_owner_containment_padding_and_thirty_second_cap(self):
        owners = [{"id": 1, "start": 0., "end": 25.},
                  {"id": 2, "start": 25., "end": 53.},
                  {"id": 3, "start": 53., "end": 60.}]
        crops = la.plan_crops(owners, 60*16000)
        self.assertEqual(len(crops), 3)
        for owner, crop in zip(owners, crops):
            self.assertLessEqual(crop["crop_start_frame"], owner["start"]*16000)
            self.assertGreaterEqual(crop["crop_end_frame"], owner["end"]*16000)
            self.assertLessEqual(crop["crop_end_frame"]-crop["crop_start_frame"], 30*16000)
        self.assertEqual(crops[0]["crop_start_frame"], 0)
        self.assertEqual(crops[-1]["crop_end_frame"], 60*16000)

    def test_deep_only_two_views_each_and_explicit_owners(self):
        owners = [{"id": i+1, "start": i*20., "end": (i+1)*20.} for i in range(20)]
        crops = la.plan_crops(owners, 400*16000, mode="deep",
                             flagged_ids=list(range(1,21)), masking_ids=[1,3])
        self.assertEqual(len(crops)*2, 80)
        self.assertEqual([c["kind"] for c in crops[:2]], ["bandit_dialogue", "bandit_residual"])
        self.assertEqual([c["kind"] for c in crops[2:4]], ["raw_early", "raw_late"])
        with self.assertRaises(ValueError):
            la.plan_crops(owners, 400*16000, flagged_ids=[1])
        with self.assertRaises(ValueError):
            la.plan_crops(owners, 400*16000, mode="deep", flagged_ids=[1], masking_ids=[2])

    def test_vad_moves_boundaries_without_removing_owner_sound(self):
        owner = {"id":1, "start": 10., "end": 25.}
        speech = [{"start": 9*16000, "end": 26*16000}]
        crops = la.plan_crops([owner], 40*16000, mode="deep", flagged_ids=[1], speech=speech)
        for crop in crops:
            self.assertLessEqual(crop["crop_start_frame"], 10*16000)
            self.assertGreaterEqual(crop["crop_end_frame"], 25*16000)
            self.assertLessEqual(crop["crop_end_frame"]-crop["crop_start_frame"], 30*16000)

    def test_bad_owner_types_and_semantic_fields_rejected(self):
        for owner in [{"id": True, "start": 0., "end": 1.},
                      {"id": 1, "start": float("nan"), "end": 1.},
                      {"id": 1, "start": 0., "end": 31.},
                      {"id": 1, "start": 0., "end": 1., "text": "synthetic"}]:
            with self.assertRaises(ValueError):
                la.plan_crops([owner], 40*16000)

    def test_millisecond_srt_tail_rounding_is_clamped_without_frame_loss(self):
        duration = 64003 / 16000
        rows = la.validate_owners([{"id":1,"start":0.,"end":round(duration,3)}],duration)
        crops = la.plan_crops(rows,64003)
        self.assertEqual(crops[0]["crop_end_frame"],64003)

    def test_duplicate_boundary_views_are_explicit(self):
        rows = la.plan_crops([{"id":1,"start":0.,"end":30.}], 30*16000,
                            mode="deep", flagged_ids=[1])
        self.assertFalse(rows[0]["duplicate_geometry"])
        self.assertTrue(rows[1]["duplicate_geometry"])


class LifecycleTests(unittest.TestCase):
    def test_actual_go_idle_payload_and_malformed_status(self):
        self.assertIsNone(la._backend_identity({"state":"idle","refcount":0}))
        self.assertIsNone(la._backend_identity({"state":"idle","refcount":0,"loaded_model":None}))
        self.assertEqual(la._backend_identity({"state":"ready","refcount":0,"loaded_model":"actual"}),"actual")
        for value in ({}, {"state":"idle"}, {"state":"idle","refcount":1},
                      {"state":"idle","refcount":False}, {"state":"loading","refcount":0},
                      {"state":"loading","refcount":0,"loaded_model":None}):
            with self.assertRaises(RuntimeError): la._backend_identity(value)

    def test_previously_unloaded_state_stays_unloaded_on_body_failure(self):
        calls = []
        def request(url, payload=None):
            calls.append((url,payload))
            return {"state":"idle","refcount":0}
        with tempfile.TemporaryDirectory() as tmp, patch.object(la,"_request_json",side_effect=request):
            path=Path(tmp)/"life.json"
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                with la._gpu_lifecycle("http://127.0.0.1:8089/admin",path):
                    raise RuntimeError("synthetic")
            receipt=la._read(path)
            self.assertTrue(receipt["restored"])
            self.assertEqual(receipt["body_error_type"],"RuntimeError")
            self.assertFalse(any(url.endswith("/load") for url,_ in calls))

    def test_restores_actual_original_model(self):
        state={"model":"original-model"}
        calls=[]
        def request(url,payload=None):
            calls.append(url)
            if url.endswith("/unload"): state["model"]=None
            if url.endswith("/load"):
                state["model"]=payload["model"]
                return {"loaded":state["model"]}
            return {"loaded_model":state["model"],"state":"ready","refcount":0} if state["model"] else {"state":"idle","refcount":0}
        with tempfile.TemporaryDirectory() as tmp, patch.object(la,"_request_json",side_effect=request):
            with la._gpu_lifecycle("http://127.0.0.1:8089/admin",Path(tmp)/"life.json"):
                self.assertIsNone(state["model"])
            self.assertEqual(state["model"],"original-model")
            self.assertEqual(sum(url.endswith("/load") for url in calls),1)

    def test_no_restore_over_unconfirmed_live_worker(self):
        calls=[]
        def request(url,payload=None):
            calls.append(url)
            return {"loaded_model":"original","state":"ready","refcount":0} if len(calls)==1 else {"state":"idle","refcount":0}
        with tempfile.TemporaryDirectory() as tmp, patch.object(la,"_request_json",side_effect=request):
            path=Path(tmp)/"life.json"
            with self.assertRaisesRegex(RuntimeError,"termination"):
                with la._gpu_lifecycle("http://127.0.0.1:8089/admin",path) as receipt:
                    receipt["owned_workers_stopped"]=False
            self.assertFalse(la._read(path)["restored"])
            self.assertFalse(any(url.endswith("/load") for url in calls))


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.media=self.root/"synthetic.wav"
        # Unequal stereo channels verify native channel preservation.
        t=np.arange(4*44100,dtype=np.float32)/44100
        sf.write(self.media,np.stack([.01*np.sin(2*np.pi*200*t),.02*np.sin(2*np.pi*300*t)],axis=1),
                 44100,subtype="FLOAT")
        self.owners=[{"id":1,"start":0.,"end":2.},{"id":2,"start":2.,"end":4.}]

    def tearDown(self):
        self.temp.cleanup()

    def fake_models(self,*args):
        return {"available":True,"model_dir":str(self.root),"files":[]}

    @staticmethod
    @contextmanager
    def fake_gpu(admin,path):
        receipt={"original_loaded_model":None,"owned_workers_stopped":True,"restored":True}
        yield receipt
        la._write(path,receipt)

    def fake_launch(self,spec,folder,python,**kwargs):
        self.assertEqual(spec["worker"],"asr")
        folder.mkdir()
        for item in spec["observations"]:
            la._write(Path(item["receipt_path"]),{"observation":item,"text":"synthetic native observation",
                "status":"ok","retries":0,"model_identity_sha256":la._hash(spec["model"]),
                "worker_sha256":spec["worker_sha256"],"runtime_identity_sha256":la._hash(spec["runtime"]),"native":{"logical_requests":1,"normal_stop":True}})

    def test_native_master_rate_channel_float_and_stream_identity(self):
        manifest=la.preserve_master(self.media,self.root/"master")
        self.assertEqual(manifest["audio"]["sample_rate"],44100)
        self.assertEqual(manifest["audio"]["channels"],2)
        self.assertEqual(manifest["audio"]["subtype"],"FLOAT")
        self.assertEqual(manifest,la.preserve_master(self.media,self.root/"master"))
        with self.assertRaises(ValueError):
            la.preserve_master(self.media,self.root/"master",audio_stream=1)

    def test_prepare_then_once_execute_and_immutable_pool(self):
        out=self.root/"acq"
        with patch.object(la,"_model_pins",side_effect=self.fake_models), \
             patch.object(la,"_launch",side_effect=self.fake_launch) as launch, \
             patch.object(la,"_gpu_lifecycle",side_effect=self.fake_gpu):
            prepared=la.acquire(self.media,self.owners,out)
            self.assertEqual(prepared["maximum_asr_requests"],4)
            self.assertEqual(launch.call_count,0)
            result=la.acquire(self.media,self.owners,out,execute=True)
            self.assertTrue(result["complete"])
            self.assertEqual(len(result["observations"]),4)
            self.assertEqual(launch.call_count,2)
            again=la.acquire(self.media,self.owners,out,execute=True)
            self.assertEqual(again,result)
            self.assertEqual(launch.call_count,2)
            changed=copy.deepcopy(result)
            changed["observations"][0]["text"]="changed"
            with self.assertRaises(ValueError):
                la.validate_evidence(changed)

    def test_missing_models_are_explicit_not_fabricated(self):
        with patch.object(la,"_model_pins",return_value={"available":False,"reason":"synthetic"}), \
             patch.object(la,"_launch") as launch:
            result=la.acquire(self.media,self.owners,self.root/"acq",execute=True)
        self.assertEqual(launch.call_count,0)
        self.assertFalse(result["complete"])
        self.assertEqual(result["counts"]["unavailable"],4)
        self.assertTrue(all(o["text"]=="" for o in result["observations"]))

    def test_started_acquisition_cannot_retry(self):
        out=self.root/"acq"
        with patch.object(la,"_model_pins",side_effect=self.fake_models):
            la.acquire(self.media,self.owners,out)
            la._write(out/"execution-started.json",{"synthetic":True})
            with patch.object(la,"_launch") as launch:
                with self.assertRaisesRegex(RuntimeError,"cannot be rerun"):
                    la.acquire(self.media,self.owners,out,execute=True)
                self.assertEqual(launch.call_count,0)

    def test_worker_failure_keeps_each_native_receipt(self):
        with patch.object(la,"_model_pins",side_effect=self.fake_models), \
             patch.object(la,"_launch",side_effect=RuntimeError("synthetic")), \
             patch.object(la,"_gpu_lifecycle",side_effect=self.fake_gpu):
            result=la.acquire(self.media,self.owners,self.root/"acq",execute=True)
        self.assertEqual(result["counts"]["unavailable"],4)
        self.assertTrue(all(Path(o["receipt_path"]).is_file() for o in result["observations"]))

    def test_raw_waveform_tampering_invalidates_pool(self):
        with patch.object(la,"_model_pins",side_effect=self.fake_models), \
             patch.object(la,"_launch",side_effect=self.fake_launch), \
             patch.object(la,"_gpu_lifecycle",side_effect=self.fake_gpu):
            result=la.acquire(self.media,self.owners,self.root/"acq",execute=True)
        wav=Path(result["observations"][0]["provenance"]["wav_path"])
        wav.chmod(0o644)
        with wav.open("ab") as stream: stream.write(b"changed")
        with self.assertRaises(ValueError):
            la.validate_evidence(result)


class BanditWorkerTests(unittest.TestCase):
    def test_native_all_stems_and_residual_preserve_frames_without_gpu(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); checkpoint=root/"checkpoint.ckpt"; checkpoint.write_bytes(b"synthetic")
            source=root/"source.py"; source.write_text("# synthetic")
            wav=root/"original48.wav"
            audio=np.linspace(-.2,.2,4800,dtype=np.float32)
            sf.write(wav,audio,48000,subtype="FLOAT")
            loads=[]
            class Tensor:
                def __init__(self,value): self.value=value
                def __getitem__(self,key): return Tensor(self.value[key])
                def cuda(self): return self
                def detach(self): return self
                def cpu(self): return self
                def numpy(self): return self.value
            class Model:
                def __init__(self,**kwargs): pass
                def load_state_dict(self,state,strict):
                    loads.append((state,strict))
                def eval(self): return self
                def cuda(self): return self
            class Handler:
                def __init__(self,**kwargs): pass
                def __call__(self,x,model):
                    return {"estimates":{name:{"audio":Tensor(x.value*fraction)} for name,fraction in
                                         (("dialogue",.5),("music",.3),("effects",.2))}}
            native=SimpleNamespace(Bandit=Model,Handler=Handler)
            torch=SimpleNamespace(load=lambda *a,**k:{"state_dict":{"model.weight":"native","metric.buffer":"training"}},
                from_numpy=Tensor,inference_mode=nullcontext)
            manifest={"version":1,"source_dir":str(root),"checkpoint_path":str(checkpoint),
                "checkpoint_sha256":la.file_hash(checkpoint),"source_code_sha256":{str(source):la.file_hash(source)},
                "sample_rate":48000,"channels":1,"chunk_seconds":8.,"hop_seconds":1.,"batch_size":1,
                "native_model_import":"synthetic_bandit:Bandit","native_handler_import":"synthetic_bandit:Handler",
                "model_kwargs":{},"handler_kwargs":{},"dialogue_stem":"dialogue"}
            output=root/"worker"; output.mkdir()
            spec={"manifest":manifest,"output_dir":str(output),
                  "crops":[{"input":la._pin(wav),"output_dir":str(root/"stems")}]}
            before=list(sys.path)
            try:
                with patch.dict(sys.modules,{"torch":torch,"synthetic_bandit":native}):
                    la._bandit_worker(spec)
            finally:
                sys.path[:]=before
            self.assertEqual(loads,[({"weight":"native"},True)])
            stems=la._read(root/"stems"/"stems.json")
            self.assertEqual(set(stems["stems"]),{"dialogue","music","effects"})
            residual,rate=sf.read(stems["residual"]["path"],dtype="float32")
            self.assertEqual(rate,48000)
            self.assertEqual(len(residual),len(audio))
            np.testing.assert_allclose(residual,audio*.5,atol=1e-7)
            self.assertEqual(la._read(output/"native-load.json")["nonmodel_state_keys"],{"metric.buffer":"str"})


class InterpreterTests(unittest.TestCase):
    def test_venv_symlink_is_not_resolved_to_system_python(self):
        path = la.PROJECT / ".venv/bin/python"
        self.assertTrue(path.is_file())
        self.assertEqual(la._interpreter_path(path), str(path))
        self.assertNotEqual(la._interpreter_path(path), str(path.resolve()))

    def test_direct_script_venv_imports_from_unrelated_working_directory(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            python = la._interpreter_path(la.PROJECT / ".venv/bin/python")
            run = subprocess.run([python, str(Path(la.__file__).resolve()), "--preflight-asr-imports"],
                                 cwd=tmp, check=True, capture_output=True, text=True, timeout=60)
            result=json.loads(run.stdout)
        self.assertEqual(result["prefix"],str(la.PROJECT/".venv"))
        self.assertNotEqual(result["prefix"],result["base_prefix"])
        self.assertTrue(result["production_imports_ready"])
        self.assertEqual(result["model_loads"],0)
        self.assertEqual(result["forwards"],0)


if __name__=="__main__":
    unittest.main()
