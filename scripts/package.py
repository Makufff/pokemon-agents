"""Phase 5: Build submission.tar.gz with only the files Kaggle needs.

Usage:
    python scripts/package.py [--fp16]

--fp16: quantize model weights to float16 before packaging (halves file size).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

# Directories/files to include in the submission (relative to project root)
_INCLUDE_DIRS = ['cg', 'core', 'model', 'search', 'train']
_INCLUDE_FILES = ['main.py', 'deck.csv']
_MODEL_CANDIDATES = ['model_phase2.pt', 'model.pt']

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_TAR = os.path.join(PROJECT_ROOT, 'submission.tar.gz')


def find_model() -> str | None:
    for name in _MODEL_CANDIDATES:
        path = os.path.join(PROJECT_ROOT, name)
        if os.path.exists(path):
            return path
    return None


def quantize_fp16(src_path: str, dst_path: str) -> None:
    """Save model weights as float16 for smaller size. Loads as float32 on inference."""
    import torch
    ckpt = torch.load(src_path, map_location='cpu', weights_only=False)
    fp16_model = {k: v.half() if v.is_floating_point() else v
                  for k, v in ckpt['model'].items()}
    torch.save({'model': fp16_model, **{k: v for k, v in ckpt.items() if k != 'model'}},
               dst_path)
    orig_mb = os.path.getsize(src_path) / 1e6
    new_mb = os.path.getsize(dst_path) / 1e6
    print(f'  fp16: {orig_mb:.1f} MB → {new_mb:.1f} MB')


def build_tar(use_fp16: bool = False) -> None:
    model_src = find_model()
    if model_src is None:
        print('WARNING: no model checkpoint found — submission will use random play')

    with tempfile.TemporaryDirectory() as tmp:
        print(f'Staging in {tmp}')

        # Copy Python packages
        for d in _INCLUDE_DIRS:
            src = os.path.join(PROJECT_ROOT, d)
            dst = os.path.join(tmp, d)
            shutil.copytree(src, dst,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))

        # Copy top-level files
        for f in _INCLUDE_FILES:
            src = os.path.join(PROJECT_ROOT, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(tmp, f))

        # Model checkpoint
        if model_src is not None:
            model_name = os.path.basename(model_src)
            dst_model = os.path.join(tmp, model_name)
            if use_fp16:
                print(f'Quantizing {model_name} to fp16...')
                quantize_fp16(model_src, dst_model)
                # Also update the fallback path name in the archive
                # main.py already tries model_phase2.pt / model.pt, so name is preserved
            else:
                shutil.copy2(model_src, dst_model)
                print(f'  Copied {model_name} ({os.path.getsize(model_src)/1e6:.1f} MB)')

        # Build tar
        print(f'Writing {OUTPUT_TAR}')
        with tarfile.open(OUTPUT_TAR, 'w:gz') as tar:
            for entry in sorted(os.listdir(tmp)):
                tar.add(os.path.join(tmp, entry), arcname=entry)

    size_mb = os.path.getsize(OUTPUT_TAR) / 1e6
    print(f'Done: {OUTPUT_TAR} ({size_mb:.2f} MB)')

    # List contents
    print('\nArchive contents:')
    with tarfile.open(OUTPUT_TAR) as tar:
        for m in tar.getmembers():
            if not m.name.endswith(('.pyc', '.pyo')):
                print(f'  {m.name}  ({m.size / 1e3:.1f} KB)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true',
                        help='Quantize model to float16')
    args = parser.parse_args()
    build_tar(use_fp16=args.fp16)
