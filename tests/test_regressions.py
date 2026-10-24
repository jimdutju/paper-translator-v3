"""Regressions from real-paper comparisons; no network or detector required."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pymupdf
import quality


class QualityTests(unittest.TestCase):
    def test_missing_page_is_not_clean(self):
        rows = [{"page": 1, "role": "body", "bbox": [0, 0, 500, 500]}]
        self.assertIn("missing pages: 2", quality.transcription_warnings(rows, 2)[0])

    def test_formula_identity_including_duplicates(self):
        self.assertEqual([], quality.translation_warnings("", "$x$ 和 $x$", ["x", "x"]))
        self.assertIn("inline math changed or missing",
                      quality.translation_warnings("", "$x$", ["x", "x"]))

    def test_missing_translation_is_reported(self):
        self.assertEqual(["missing translation"], quality.translation_warnings("Some text", ""))

    def test_invisible_page_is_not_clean(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "blank.pdf"
            doc = pymupdf.open(); doc.new_page(); doc.save(path); doc.close()
            self.assertEqual([1], quality.audit_pdf(path)["blank_pages"])

    def test_skip_requires_clean_record_and_matching_hash(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sample.pdf"
            doc = pymupdf.open(); p = doc.new_page()
            p.insert_text((72, 72), "A readable document with sufficient text.")
            doc.save(path); doc.close()
            self.assertFalse(quality.record_is_clean(path, "sample"))
            rec = quality.write_record("sample", path, [], 0)
            self.assertEqual("checks_passed", rec["verdict"])
            self.assertTrue(quality.record_is_clean(path, "sample"))
            with path.open("ab") as f: f.write(b"\n% changed\n")
            self.assertFalse(quality.record_is_clean(path, "sample"))

    def test_warning_record_is_not_reused(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sample.pdf"
            doc = pymupdf.open(); p = doc.new_page()
            p.insert_text((72, 72), "A readable document with sufficient text.")
            doc.save(path); doc.close()
            quality.write_record("sample", path, ["  ! missing translation p1"], 0)
            self.assertFalse(quality.record_is_clean(path, "sample"))

    def test_batch_warning_returns_failure_exit_status(self):
        import translate
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td, \
             patch.object(sys, "argv", ["translate.py", "sample", "-o", td]), \
             patch.object(translate, "say"), \
             patch.object(translate, "run_one", return_value=(
                 "ok", "sample", str(Path(td)/"sample.pdf"),
                 (1, 0, ["! missing translation p1"], 0))):
            self.assertEqual(1, translate.main())


class DefinitionListTests(unittest.TestCase):
    def test_invalid_cached_math_is_retranslated_selectively(self):
        import trans
        from unittest.mock import patch
        bad, good = "a ⟦M1⟧ and ⟦M2⟧", "another ⟦M1⟧"
        with tempfile.TemporaryDirectory() as td, \
             patch.object(trans, "CACHE", str(Path(td)/"cache.json")), \
             patch.object(trans, "GLOSS", str(Path(td)/"gloss.json")):
            Path(trans.CACHE).write_text(json.dumps({trans._hash(bad): "漏掉 ⟦M1⟧",
                                                     trans._hash(good): "另一个 ⟦M1⟧"}), encoding="utf-8")
            tr = trans.Translator(verbose=False)
            with patch.object(tr, "paper_glossary", return_value=[]), \
                 patch.object(tr, "_batch", return_value=["保留 ⟦M1⟧ 和 ⟦M2⟧"]) as batch:
                out = tr.translate([{"en": bad, "keep": ["x", "y"]},
                                    {"en": good, "keep": ["z"]}], "sample", "sample")
            self.assertEqual(["保留 $x$ 和 $y$", "另一个 $z$"], out)
            self.assertEqual(1, len(batch.call_args.args[0]))

    def test_cached_translation_must_keep_every_math_slot(self):
        import trans
        self.assertTrue(trans._math_slots_valid("前 ⟦M1⟧ 个 ⟦M2⟧", "first ⟦M1⟧ of ⟦M2⟧"))
        self.assertFalse(trans._math_slots_valid("前 ⟦M1⟧ 个", "first ⟦M1⟧ of ⟦M2⟧"))
        self.assertFalse(trans._math_slots_valid("⟦M1⟧ ⟦M1⟧", "⟦M1⟧"))

    def test_definition_lines_remain_independent(self):
        import vlm_yolo
        source = {"page": 4, "role": "formula", "bbox": [100, 100, 900, 300],
                  "text": "$x$ = vector of active weights\n$y$ = vector of stock returns"}
        rows = vlm_yolo._split_display([source])
        self.assertEqual(2, len(rows))
        self.assertTrue(all(b.get("_definition") and b["role"] == "body" for b in rows))
        self.assertEqual(["$x$ = vector of active weights", "$y$ = vector of stock returns"],
                         [b["text"] for b in rows])

    def test_pure_display_stays_math(self):
        import vlm_yolo
        rows = vlm_yolo._split_display([{"page": 1, "role": "formula", "bbox": [0, 0, 500, 200],
                                        "text": "$x=y$\n$z=1$"}])
        self.assertTrue(all(b["role"] == "formula" and not b.get("_definition") for b in rows))


if __name__ == "__main__":
    unittest.main()
