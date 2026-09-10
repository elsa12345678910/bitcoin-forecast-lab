"""Compatibility entry point for the canonical root-level Streamlit app."""

from pathlib import Path
import runpy


runpy.run_path(Path(__file__).with_name("app.py"), run_name="__main__")
