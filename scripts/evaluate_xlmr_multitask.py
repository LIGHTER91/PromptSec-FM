#!/usr/bin/env python3
"""Evaluate one compatible selected XLM-R checkpoint on official PolicyBench splits."""

from __future__ import annotations

import sys

from train_xlmr_multitask import main

if __name__ == "__main__":
    raise SystemExit(main([*sys.argv[1:], "--evaluate-only"]))
