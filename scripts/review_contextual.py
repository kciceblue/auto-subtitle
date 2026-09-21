"""Prepare or explicitly execute one immutable contextual subtitle scalar review."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from src.contextual_review import review, validate_receipt


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--validate',action='store_true')
    args=parser.parse_args()
    try:
        result=validate_receipt(args.output_dir,args.manifest) if args.validate else review(
            args.manifest,args.output_dir,execute=args.execute)
        keys=('prepared','executed','score','contextual_usability_pass','whole_text_read','confidence',
              'dispatch_id','reviewed_date','started_utc','finished_utc','seconds','output_dir')
        print(json.dumps({key:result[key] for key in keys if key in result},ensure_ascii=False))
        return 0
    except Exception as error:
        print(json.dumps({'error_type':type(error).__name__,'executed':args.execute}))
        return 1


if __name__=='__main__':
    raise SystemExit(main())
