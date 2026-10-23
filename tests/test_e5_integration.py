import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
import warnings
from scripts import run_e5 as e5


class E5Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        X, y = make_classification(n_samples=250, n_features=150,
                                  n_informative=10, random_state=123)
        self.X, self.y = X, y
        frame = pd.DataFrame(X, columns=[str(i) for i in range(150)])
        frame.insert(0, "id", np.arange(250))
        frame["target"] = y
        self.train = self.root / "train.csv"
        frame.to_csv(self.train, index=False)
        rows = []
        self.records = []
        for seed in range(10):
            for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
                self.records.append(dict(repeat=seed, split_seed=seed, fold=fold,
                                         train_indices=tr.tolist(), val_indices=va.tolist()))
                rows.extend(dict(repeat=seed, split_seed=seed, fold=fold, row_index=int(i)) for i in va)
        self.splits = self.root / "splits.csv"
        pd.DataFrame(rows).to_csv(self.splits, index=False)
        self.refs = self.root / "refs.csv"
        pd.DataFrame([dict(repeat=r, split_seed=r, C=c, mean_auc=.8 if c == 1 else .79,
                           n_folds=5, e3_leader_auc=.77) for c in [.1, 1.] for r in range(10)]).to_csv(self.refs,index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, destination):
        return ["--train", str(self.train), "--splits", str(self.splits),
                "--references", str(self.refs), "--output", str(destination),
                "--confirm-reference-alignment"]

    def test_data_and_csv_splits(self):
        df, X, y, cols = e5.load_data(self.train)
        s = e5.load_splits(self.splits, y)
        self.assertEqual(X.shape, (250, 150))
        self.assertEqual(len(s), 2500)
        self.assertNotIn("id", cols)
        self.assertNotIn("target", cols)

    def test_json_equals_csv_splits(self):
        path = self.root / "splits.json"
        path.write_text(json.dumps(self.records))
        pd.testing.assert_frame_equal(e5.load_splits(path, self.y), e5.load_splits(self.splits,self.y))

    def test_overlap_rejected(self):
        self.records[0]["train_indices"].append(self.records[0]["val_indices"][0])
        path = self.root / "bad.json"
        path.write_text(json.dumps(self.records))
        with self.assertRaises(ValueError):
            e5.load_splits(path, self.y)

    def test_reference_duplicate_rejected(self):
        df = pd.read_csv(self.refs)
        pd.concat([df, df.iloc[:1]]).to_csv(self.refs,index=False)
        with self.assertRaises(ValueError):
            e5.load_references(self.refs)

    def test_fold_score_and_local_scaling(self):
        rec = self.records[0]
        tr, va = np.array(rec["train_indices"]), np.array(rec["val_indices"])
        meta = dict(experiment_id="test", repeat=0, split_seed=0, fold=0)
        diag, preds = e5.fit_fold(self.X, self.y, np.arange(250), tr, va, 1., "scale", meta)
        self.assertTrue(diag["valid"])
        model = e5.make_pipeline(1., "scale").fit(self.X[tr], self.y[tr])
        np.testing.assert_allclose(model.named_steps["scaler"].mean_, self.X[tr].mean(axis=0))
        self.assertEqual(diag["auc"], roc_auc_score(self.y[va],model.decision_function(self.X[va])))
        np.testing.assert_allclose([r["decision_score"] for r in preds],model.decision_function(self.X[va]))

    def test_failed_fit_not_zero_auc(self):
        tr = np.array(self.records[0]["train_indices"])
        va = np.array(self.records[0]["val_indices"])
        with patch.object(e5.Pipeline, "fit", side_effect=RuntimeError("injected failure")):
            diag, preds = e5.fit_fold(self.X, self.y, np.arange(250), tr, va, 1., "scale", {})
        self.assertFalse(diag["valid"])
        self.assertIsNone(diag["auc"])
        self.assertFalse(preds)
        self.assertIn("injected failure",diag["error"])

    def test_convergence_warning_excludes_fold(self):
        original = e5.Pipeline.fit
        def warned_fit(obj, *args, **kwargs):
            result = original(obj, *args, **kwargs)
            warnings.warn("injected warning", ConvergenceWarning)
            return result
        tr = np.array(self.records[0]["train_indices"])
        va = np.array(self.records[0]["val_indices"])
        with patch.object(e5.Pipeline, "fit", warned_fit):
            diag, _ = e5.fit_fold(self.X,self.y,np.arange(250),tr,va,1.,"scale",{})
        self.assertTrue(diag["convergence_warning"])
        self.assertFalse(diag["valid"])

    def test_dry_run_and_no_overwrite(self):
        dest = self.root / "dry"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(e5.main(self.args(dest)+["--dry-run"]),0)
        self.assertFalse((dest/"e5_fold_diagnostics.csv").exists())
        with self.assertRaises(ValueError):
            e5.main(self.args(dest))

    def test_full_300_fit_integration(self):
        dest = self.root / "results"
        with contextlib.redirect_stdout(io.StringIO()):
            result = e5.main(self.args(dest))
        self.assertEqual(result,0)
        folds = pd.read_csv(dest/"e5_fold_diagnostics.csv")
        comparison = pd.read_csv(dest/"e5_comparison.csv")
        repeats = pd.read_csv(dest/"e5_paired_repeat_scores.csv")
        self.assertEqual(len(folds),300)
        self.assertTrue(folds["valid"].all())
        self.assertEqual(len(comparison),6)
        self.assertEqual(len(repeats),60)
        for directory in dest.glob("e5_rbf_svm_*"):
            scores = pd.read_csv(directory/"validation_decision_scores.csv")
            self.assertEqual(len(scores),2500)
        for _, r in repeats.iterrows():
            selected = folds[(folds["experiment_id"]==r["experiment_id"]) & (folds["repeat"]==r["repeat"])]
            self.assertAlmostEqual(r["mean_auc"],selected["auc"].mean())
            self.assertAlmostEqual(r["difference_vs_e4_leader"],r["mean_auc"]-.8)
        decision = json.loads((dest/"e5_decision.json").read_text())
        self.assertIn(decision["e5_leader"],comparison["experiment_id"].tolist())
        self.assertEqual(json.loads((dest/"run_manifest.json").read_text())["completed_fits"],300)


if __name__ == "__main__":
    unittest.main(verbosity=2)
