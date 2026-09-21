from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


LANG_CODES: dict[str, str] = {
    "simplified chinese": "zh",
    "traditional chinese": "zh-Hant",
    "chinese": "zh",
    "japanese": "ja",
    "english": "en",
    "korean": "ko",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "portuguese": "pt",
    "russian": "ru",
    "thai": "th",
    "vietnamese": "vi",
    "indonesian": "id",
    "arabic": "ar",
}

MEDIA_EXTENSIONS: set[str] = {
    ".mkv", ".mp4", ".webm", ".avi", ".mov", ".flv", ".wmv",
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma",
}


@dataclass
class TranscribeConfig:
    """Configuration for the transcription pipeline.

    For batch mode (no input_file), set input_file to None
    and provide input_dir / output_dir instead.
    """

    input_file: Path | None = None
    output_file: Path | None = None
    input_dir: Path = field(default_factory=lambda: Path("input"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    language: str = "auto"
    model: str = "large-v3"
    compute_type: str = "float16"
    beam_size: int = 5
    no_demucs: bool = False
    keep_temp: bool = False
    cached_audio: Path | None = None
    verbose: bool = False
    # ASR hotwords: explicit word list passed to faster-whisper to bias
    # transcription toward known terminology (character names, domain terms).
    hotwords: list[str] = field(default_factory=list)
    hotword_mode: str = "model"  # "explicit" uses files; "none" disables ASR bias
    ensemble_source: str = "consensus"  # optional original-audio Qwen or NeMo preference
    nemo_model: Path | None = None  # local NeMo archive, required only for NeMo source
    nemo_python: Path | None = None  # executable inside an isolated NeMo environment
    qwen_asr_hotwords: bool = True
    asr_window_seconds: float = 20.0  # core duration; 0.8s context on each edge
    source_pause_units: bool = False  # opt-in token-preserving pause split before translation
    # File(s) containing hotword candidates (one per line). Merged with
    # LLM-generated hotwords from README/context files.
    hotwords_file: Path | None = None
    # Context files (e.g. README.txt) used to LLM-generate hotwords before ASR.
    context_files: list[Path] = field(default_factory=list)
    # Content-kind per context file (path-string → "synopsis"|"script"),
    # produced by src/context_scan (content-aware discovery).
    context_kinds: dict[str, str] | None = None
    # LLM endpoint + payload for hotword generation / proofreading (reuses
    # the translation endpoint).
    hotword_endpoint: str | None = None
    hotword_extra_payload: dict | None = None

    # VAD parameters
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 500
    vad_speech_pad_ms: int = 400

    # llama-warden GPU coordination: whisper (~4 GB) and the resident 27B
    # translator (~29 GB) cannot share the 32 GB card, so before ASR we evict
    # the warden LLM when VRAM headroom is short (src/warden.py). Disable
    # with --no-warden-unload if warden is actively serving other requests.
    warden_admin_url: str = "http://127.0.0.1:8089/admin"
    unload_warden_before_asr: bool = True

    def __post_init__(self) -> None:
        if self.hotword_mode not in {"model", "explicit", "none"}:
            raise ValueError(f"Unknown hotword mode: {self.hotword_mode}")
        import math
        if (isinstance(self.asr_window_seconds, bool)
                or not isinstance(self.asr_window_seconds, (int, float))
                or not math.isfinite(self.asr_window_seconds)
                or not 8.0 <= self.asr_window_seconds <= 28.0):
            raise ValueError("ASR window core must be finite and between 8 and 28 seconds")
        if self.ensemble_source not in {"consensus", "qwen", "nemo"}:
            raise ValueError(f"Unknown ensemble source policy: {self.ensemble_source}")
        if self.ensemble_source == "nemo":
            import os
            if self.language not in {"ja", "auto", None}:
                raise ValueError("The NeMo source model supports Japanese audio only")
            if self.nemo_model is None or self.nemo_python is None:
                raise ValueError("NeMo source requires --nemo-model and --nemo-python")
            self.nemo_model = Path(self.nemo_model).expanduser().absolute()
            self.nemo_python = Path(self.nemo_python).expanduser().absolute()
            if not self.nemo_model.is_file():
                raise FileNotFoundError(f"NeMo model archive not found: {self.nemo_model}")
            if not self.nemo_python.is_file() or not os.access(self.nemo_python, os.X_OK):
                raise ValueError(f"NeMo Python is not an executable file: {self.nemo_python}")
        elif self.nemo_model is not None or self.nemo_python is not None:
            raise ValueError("NeMo paths require --ensemble-source nemo")
        if self.input_file is not None:
            self.input_file = Path(self.input_file)
            if not self.input_file.exists():
                raise FileNotFoundError(f"Input file not found: {self.input_file}")

        if self.output_file is not None:
            self.output_file = Path(self.output_file)

        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        if self.hotwords_file is not None:
            self.hotwords_file = Path(self.hotwords_file)
        self.context_files = [Path(p) for p in self.context_files]

        if self.language == "auto":
            self.language = None  # type: ignore[assignment]

    def resolve_output(self, input_path: Path) -> Path:
        """Determine output .srt path for a given input file.

        Preserves directory structure relative to input_dir.
        e.g. input/series/ep01.mkv → output/series/ep01.srt
        """
        if self.output_file is not None:
            return self.output_file
        try:
            rel = input_path.relative_to(self.input_dir)
        except ValueError:
            rel = Path(input_path.name)
        return self.output_dir / rel.with_suffix(".srt")

    def collect_input_files(self) -> list[Path]:
        """Recursively collect media files from input_dir, sorted by path."""
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        files = sorted(
            f for f in self.input_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS
        )
        return files

    @property
    def vad_parameters(self) -> dict:
        return {
            "threshold": self.vad_threshold,
            "min_silence_duration_ms": self.vad_min_silence_ms,
            "speech_pad_ms": self.vad_speech_pad_ms,
        }


@dataclass
class TranslateConfig:
    """Configuration for SRT translation via LLM endpoint."""

    input_srt: Path | None = None
    output_srt: Path | None = None
    endpoint: str = "http://127.0.0.1:5000/v1/chat/completions"
    source_lang: str = "Japanese"
    target_lang: str = "Simplified Chinese"
    chunk_size: int = 10
    timeout: int = 300
    retries: int = 2
    # Token budget per LLM request, always sent explicitly. Relying on the
    # server default (llama-server --predict) is unsafe with a reasoning
    # model: it can spend the entire budget on reasoning_content and return
    # content="" with finish_reason="length".
    #
    # Default raised to 16384 (2026-08-30, FTDC 实测): review DeepFix chunks
    # routinely spend 12k-27k reasoning chars (~6k-13k tokens) — under the
    # old 8192 default, 10/28 requests in a full-file run would overrun and
    # trigger budget-growth retries (full re-inference + backoff sleep).
    # Streaming + EOS early-stop means max_tokens is only an upper fuse:
    # translation/proofread (thinking-off) are unaffected by the larger cap.
    max_tokens: int = 16384
    extra_payload: dict | None = None
    separate_instruction: bool = False  # opt-in system context + user batch for prefix caching
    translation_reasoning_budget: int = 0  # first draft only: 0/off, 256 or 512 tokens per thinking block
    translation_episode_context: bool = False  # complete chosen-source context, first draft only
    coherence_polish: bool = False  # optional local document-context editing before display
    coherence_recipe: Path | None = None  # explicit local multi-pass editor configuration
    vocab_file: Path | None = None
    # Context file(s) (e.g. README.txt / synopsis) injected into the instruction
    # prefix so the LLM understands the story, characters and domain
    # terminology before translating.
    context_files: list[Path] = field(default_factory=list)
    # Content-kind per context file (path-string → "synopsis"|"script"),
    # produced by src/context_scan (content-aware discovery).
    context_kinds: dict[str, str] | None = None
    # Optional meaningful media title (src/title_context.extract_title):
    # injected into translation/review context so domain terms from the
    # filename (e.g. 去勢) help the LLM pick terms the ASR garbled.
    title: str | None = None
    # Compact work context and immutable cue evidence for the new workflow.
    context_summary: str | None = None
    line_notes: dict[int, list[str]] = field(default_factory=dict)
    scene_context_lines: int = 3
    pack_scenes: bool = False
    structured_qa: bool = False  # experimental local llama.cpp JSON-schema QA
    structured_translation: bool = False  # optional exact local-ID JSON draft responses
    delta_verification: bool = False  # experimental before/after repair transactions
    pause_layout: bool = False  # experimental exact target partition at supported pauses
    source_span_repair: bool = False  # experimental fixed raw-ASR alternatives plus realignment
    entity_placeholders: bool = False  # experimental grounded temporary name markers
    entity_notes: dict[int, list[str]] = field(default_factory=dict)
    scene_max_chars: int = 2400
    telemetry_path: Path | None = None
    stage: str = "llm"
    response_guard_floor: int = 0  # structured outputs have fixed schema overhead
    # Optional dialogue script (台词台本) of the work — ground truth that
    # outranks audio arbitration. Injected into translation context with
    # script-aware sampling, and anchored into the review pass
    # (src/script_align.py). Explicit --script wins; otherwise the first
    # "script"-kind context file is used automatically.
    script_file: Path | None = None
    # After translation, run a whole-file proofread pass that fixes
    # consistency/alignment issues while keeping 1:1 line alignment.
    proofread: bool = False
    proofread_context_lines: int = 3  # original context lines shown to LLM
    # Semantic review pass (after proofread): thinking-enabled, evidence-based
    # gate. Only lines the model can verify are released; garbled/unsolvable
    # lines get a 存疑 flag instead of a speculative fix. Writes a
    # REVIEW-*.md report next to the translated SRT.
    review: bool = False
    review_chunk_size: int = 8
    review_context_lines: int = 3
    # Optional reference SRT (hand-corrected or previously accepted
    # translation) whose lines are matched by timestamp overlap and shown
    # to the review LLM as semantic anchors.
    reference_srt: Path | None = None
    # Optional FTDC adjudication JSON (src/adjudicate.py) — enables the fast
    # review path: triage → packed DeepFix on suspicious lines with audio
    # evidence → incremental critic (≤2 rounds).
    adjudication: Path | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        if type(self.separate_instruction) is not bool:
            raise ValueError("separate_instruction must be a boolean")
        if type(self.translation_reasoning_budget) is not int or self.translation_reasoning_budget not in (0, 256, 512):
            raise ValueError("Translation reasoning budget must be 0, 256 or 512")
        if self.translation_reasoning_budget and self.max_tokens <= self.translation_reasoning_budget:
            raise ValueError("Total max_tokens must leave final-answer room after the translation reasoning budget")
        if self.input_srt is not None:
            self.input_srt = Path(self.input_srt)
        if self.output_srt is not None:
            self.output_srt = Path(self.output_srt)
        if self.vocab_file is not None:
            self.vocab_file = Path(self.vocab_file)
            if not self.vocab_file.exists():
                raise FileNotFoundError(f"Vocab file not found: {self.vocab_file}")
        self.context_files = [Path(p) for p in self.context_files]
        if self.reference_srt is not None:
            self.reference_srt = Path(self.reference_srt)
            if not self.reference_srt.exists():
                raise FileNotFoundError(f"Reference SRT not found: {self.reference_srt}")
        if self.adjudication is not None:
            self.adjudication = Path(self.adjudication)
            if not self.adjudication.exists():
                raise FileNotFoundError(f"Adjudication JSON not found: {self.adjudication}")
        if self.script_file is not None:
            self.script_file = Path(self.script_file)
            if not self.script_file.exists():
                raise FileNotFoundError(f"Script file not found: {self.script_file}")
        missing_ctx = [p for p in self.context_files if not p.exists()]
        if missing_ctx:
            raise FileNotFoundError(f"Context file(s) not found: {missing_ctx}")

    @property
    def target_lang_code(self) -> str:
        """Short code for target language, e.g. 'zh', 'en'."""
        return LANG_CODES.get(self.target_lang.lower(), self.target_lang.lower())

    def resolve_translate_output(self, input_path: Path) -> Path:
        """Determine output path for translated SRT.

        Default: input_stem.{lang_code}.srt (e.g. movie.zh.srt)
        """
        if self.output_srt is not None:
            return self.output_srt
        code = self.target_lang_code
        stem = input_path.stem
        # Strip existing lang suffix if re-translating (e.g. movie.zh.srt)
        if stem.endswith(f".{code}"):
            return input_path
        return input_path.parent / f"{stem}.{code}.srt"
