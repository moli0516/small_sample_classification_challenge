"""Compatibility entry point for the relocated E5 runner."""
import runpy

if __name__ == "__main__":
    runpy.run_module("scripts.run_e5", run_name="__main__")
