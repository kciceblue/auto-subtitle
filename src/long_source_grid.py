"""Exact physical-sample tiling for unaligned long ASR observations; no audio I/O."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
import re
from src import evidence_context as ec
from src.contextual_review_contract import require
VERSION = 'long-source-grid-1'
SAMPLE_RATE = 16000
WINDOW_FRAMES = 120*SAMPLE_RATE


def _frame(value):
    match = re.fullmatch(r'(\d{2,}):(\d{2}):(\d{2}),(\d{3})', value)
    require(match is not None, 'Invalid original timestamp')
    hours, minutes, seconds, millis = map(int, match.groups())
    require(minutes < 60 and seconds < 60, 'Invalid timestamp component')
    return (((hours*60+minutes)*60+seconds)*1000+millis)*16


def build_grid(source_rows, mono_frames, sample_rate=SAMPLE_RATE):
    require(type(sample_rate) is int and sample_rate == SAMPLE_RATE, 'Fixed16k sample grid required')
    require(type(mono_frames) is int and mono_frames > 0, 'Positive physical frame count required')
    require(type(source_rows) is list and source_rows, 'Original source rows required')
    rows = [asdict(row) if is_dataclass(row) else row for row in source_rows]
    owners = []; previous = 0
    for number, row in enumerate(rows, 1):
        require(type(row) is dict and set(row) == {'index', 'ts_line', 'text'} and type(row['index']) is int
                and row['index'] == number and type(row['text']) is str and type(row['ts_line']) is str, 'Original owner shape changed')
        parts = row['ts_line'].split(' --> '); require(len(parts) == 2, 'Invalid owner interval')
        start, end = map(_frame, parts)
        require(start == previous and start < end and start < mono_frames, 'Original owners must be contiguous from zero')
        tail = max(0, end-mono_frames)
        require(tail == 0 or number == len(rows) and tail <= 2, 'Unexpected original timestamp tail')
        owners.append({'owner_id': number, 'owner_ts_line': row['ts_line'], 'owner_start_frame': start,
                       'owner_end_frame': end, 'owner_out_of_domain_frames': tail})
        previous = end
    require(previous >= mono_frames, 'Original source does not cover physical recording')
    windows = []
    for number, start in enumerate(range(0, mono_frames, WINDOW_FRAMES), 1):
        end = min(start+WINDOW_FRAMES, mono_frames); intersections = []
        for owner in owners:
            left, right = max(start, owner['owner_start_frame']), min(end, owner['owner_end_frame'])
            if left < right:
                intersections.append({**owner, 'intersection_start_frame': left, 'intersection_end_frame': right})
        require(sum(x['intersection_end_frame']-x['intersection_start_frame'] for x in intersections) == end-start,
                'Owner intersections do not partition the physical window')
        windows.append({'window_id': f'window-{number:03d}', 'crop_start_frame': start, 'crop_end_frame': end,
                        'sample_rate': SAMPLE_RATE, 'owner_intersections': intersections, 'internal_word_alignment': 'unknown'})
    return {'version': VERSION, 'source_rows_sha256': ec._hash(rows), 'original_mono_frames': mono_frames,
            'sample_rate': SAMPLE_RATE, 'window_frames': WINDOW_FRAMES, 'interval_convention': 'absolute_half_open',
            'source_owners': owners, 'windows': windows}


def validate_grid(grid, source_rows, mono_frames, sample_rate=SAMPLE_RATE):
    require(ec._hash(grid) == ec._hash(build_grid(source_rows, mono_frames, sample_rate)), 'Long grid differs from original physical source')
    return {'owners': len(grid['source_owners']), 'windows': len(grid['windows']), 'covered_frames': mono_frames,
            'owner_out_of_domain_frames': sum(x['owner_out_of_domain_frames'] for x in grid['source_owners'])}


def validate_geometry(grid):
    """Validate closed geometry without authenticating unavailable source literals."""
    require(type(grid) is dict and type(grid.get('source_rows_sha256')) is str
            and re.fullmatch('[0-9a-f]{64}', grid['source_rows_sha256']) is not None, 'Invalid source fingerprint')
    owners=grid.get('source_owners');require(type(owners) is list and owners, 'Owner geometry required')
    rows=[{'index':owner['owner_id'],'ts_line':owner['owner_ts_line'],'text':''} for owner in owners]
    expected=build_grid(rows,grid['original_mono_frames'],grid['sample_rate'])
    expected['source_rows_sha256']=grid['source_rows_sha256']
    require(ec._hash(expected)==ec._hash(grid),'Long grid structural geometry changed')
    return {'owners':len(owners),'windows':len(grid['windows']),'covered_frames':grid['original_mono_frames']}
