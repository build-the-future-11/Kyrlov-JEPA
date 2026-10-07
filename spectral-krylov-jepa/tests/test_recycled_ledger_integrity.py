"""Offline ledger counterexamples; no training, solver, or protected data use."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spectral_krylov_jepa.evaluation.recycled_policy import (
    DevelopmentResidual, development_records_from_payload, freeze_recycled_policy,
)


def fixture():
    return {
        "source_role": "development",
        "protected_outcomes_opened": False,
        "query_exact_eigensolves_performed": False,
        "records": [
            {"case_id": case, "rank": rank, "operator_applications": rank,
             "residual_relative_to_hx": value, "source_role": "development"}
            for case, values in [("dev-a", [0.20, 0.10]), ("dev-b", [0.22, 0.11])]
            for rank, value in zip([1, 2], values)
        ],
    }


def freeze(payload, **kwargs):
    return freeze_recycled_policy(development_records_from_payload(payload),
                                  candidate_ranks=[1, 2], **kwargs)


class LedgerIntegrityTests(unittest.TestCase):
    def test_valid_policy_and_order_invariance(self):
        payload = fixture()
        first = freeze(payload)
        payload["records"].reverse()
        self.assertEqual(first, freeze(payload))
        self.assertEqual(first.rank, 2)
        self.assertEqual(first.refresh_threshold, 0.11)

    def test_explicit_protected_access_cannot_be_relabelled(self):
        for field in ["protected_outcomes_opened", "query_exact_eigensolves_performed"]:
            for value in [True, 0, "false", None]:
                with self.subTest(field=field, value=value):
                    payload = fixture(); payload[field] = value
                    with self.assertRaisesRegex(ValueError, "explicitly declare"):
                        freeze(payload)

    def test_absent_boundary_declaration_is_rejected(self):
        for field in ["protected_outcomes_opened", "query_exact_eigensolves_performed"]:
            payload = fixture(); del payload[field]
            with self.assertRaisesRegex(ValueError, "explicitly declare"):
                freeze(payload)

    def test_fractional_string_or_boolean_counts_are_not_coerced(self):
        for field in ["rank", "operator_applications"]:
            for value in [1.9, 1.0, "1", True]:
                with self.subTest(field=field, value=value):
                    payload = fixture(); payload["records"][0][field] = value
                    with self.assertRaisesRegex(ValueError, "without coercion"):
                        freeze(payload)

    def test_invalid_case_id_is_not_stringified(self):
        for value in [None, 4, True, " "]:
            payload = fixture(); payload["records"][0]["case_id"] = value
            with self.assertRaisesRegex(ValueError, "case_id"):
                freeze(payload)

    def test_candidate_and_budget_counts_are_strict(self):
        records = development_records_from_payload(fixture())
        for value in [1.9, "1", True]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "without coercion"):
                freeze_recycled_policy(records, candidate_ranks=[value, 2])
            with self.subTest(budget=value), self.assertRaisesRegex(ValueError, "without coercion"):
                freeze_recycled_policy(records, candidate_ranks=[1, 2], max_operator_applications=value)

    def test_nonfinite_and_nonnumeric_residuals_are_rejected(self):
        for value in [True, "0.1", float("nan"), float("inf"), -0.1]:
            payload = fixture(); payload["records"][0]["residual_relative_to_hx"] = value
            with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                freeze(payload)

    def test_record_role_is_not_overridden_by_ledger_role(self):
        payload = fixture(); payload["records"][0]["source_role"] = "confirmatory"
        with self.assertRaisesRegex(ValueError, "development-only"):
            freeze(payload)

    def test_duplicate_and_rank_dependent_cases_remain_rejected(self):
        payload = fixture(); payload["records"].append(copy.deepcopy(payload["records"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            freeze(payload)
        payload = fixture(); payload["records"][0]["case_id"] = "different"
        with self.assertRaisesRegex(ValueError, "identical"):
            freeze(payload)

    def test_cli_hashes_exact_input_and_refuses_artifact_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "input.json"; output = Path(tmp) / "policy.json"
            raw = (json.dumps(fixture(), indent=4) + "\n").encode()
            ledger.write_bytes(raw)
            cmd = [sys.executable, str(ROOT / "scripts/18_freeze_recycled_ritz_policy.py"),
                   "--development-ledger", str(ledger), "--output", str(output),
                   "--candidate-ranks", "1", "2", "--max-operator-applications", "2"]
            env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
            first = subprocess.run(cmd, capture_output=True, env=env)
            self.assertEqual(first.returncode, 0, first.stderr.decode())
            receipt = json.loads(output.read_bytes())
            self.assertEqual(receipt["development_ledger_sha256"], hashlib.sha256(raw).hexdigest())
            before = output.read_bytes()
            again = subprocess.run(cmd, capture_output=True, env=env)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(ledger.read_bytes(), raw)
            self.assertIs(receipt["protected_outcomes_opened"], False)


if __name__ == "__main__":
    unittest.main()
