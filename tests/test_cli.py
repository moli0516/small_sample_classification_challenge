"""Unified CLI routing tests: no native imports or actual training."""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("challenge_cli_test_target",ROOT/"cli.py")
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


class cliTests(unittest.TestCase):
    def test_registry_arguments_forwarded(self):
        with patch.object(cli,"execute",return_value=0) as call:
            self.assertEqual(cli.main(["registry","cv","--config","custom.json","--dry-run"]),0)
        call.assert_called_once_with(["-m","scripts.run_registered_ensembles","cv","--config","custom.json","--dry-run"])
    def test_optuna_arguments_forwarded(self):
        with patch.object(cli,"execute",return_value=0) as call:cli.main(["lgbm-optuna","search","--dry-run"])
        call.assert_called_once_with(["-m","scripts.run_lgbm_optuna","search","--dry-run"])
    def test_child_help_forwarded(self):
        with patch.object(cli,"execute",return_value=0) as call:cli.main(["lgbm-optuna","--help"])
        call.assert_called_once_with(["-m","scripts.run_lgbm_optuna","--help"])
    def test_same_python_and_project_cwd(self):
        with patch.object(cli.subprocess,"run",return_value=type("Result",(),{"returncode":7})()) as call:
            self.assertEqual(cli.execute(["-m","pytest"]),7)
        call.assert_called_once_with([sys.executable,"-m","pytest"],cwd=ROOT,check=False)
    def test_new_tests_scope(self):
        with patch.object(cli,"execute",return_value=0) as call:cli.main(["test","--scope","new"])
        self.assertIn("tests/test_lgbm_optuna_experiment.py",call.call_args.args[0])
    def test_legacy_script_forwarding(self):
        with patch.object(cli,"execute",return_value=0) as call:cli.main(["run","run_e5b","--help"])
        call.assert_called_once_with(["-m","scripts.run_e5b","--help"])
    def test_unknown_script_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):cli.main(["run","arbitrary_script"])
    def test_plan_read_only(self):
        with patch.object(cli,"execute") as call,contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(["plan"]),0)
        self.assertIn("lgbm-optuna search --dry-run",output.getvalue());call.assert_not_called()
    def test_models_route(self):
        with patch.object(cli,"execute",return_value=0) as call:cli.main(["models"])
        call.assert_called_once_with(["-m","scripts.run_registered_ensembles","list"])
    def test_results_read_only(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root)/"decision.json").write_text('{"promote":false}')
            with patch.object(cli,"execute") as call,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["results","--run",root]),0)
            call.assert_not_called()


if __name__=="__main__":unittest.main()
