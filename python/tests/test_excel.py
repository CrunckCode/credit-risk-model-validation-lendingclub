"""Workbook tests: plain formatting, author, formula/typed separation, static conclusions, bin rule, LibreOffice reconciliation."""
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import openpyxl
import pytest

from credit_validation import reconcile_excel as rx

# ===== CONFIG (user inputs) =====
FORMULA_SHEETS = ("Score_Calc", "Deciles", "Calibration", "Checks")
MIXED_SHEETS = ("PSI", "EL")                 # typed reference columns listed in meta["typed_cols"]
STATIC_SHEETS = ("README", "Conclusions")
LIVE_PREFIX = "xl_"
MIN_CASES = 2
REL_STEP = 1e-9                              # spreadsheets treat values within ~1e-15 relative as equal
EDGE_PROBES = (-1e6, 0.0, 1.0, 2.0, 3.5, 100.0)
# ===== END CONFIG =====

needs_lo = pytest.mark.skipif(rx.find_soffice() is None, reason="LibreOffice not installed")


@pytest.fixture(scope="module")
def tmp_dir():
    with tempfile.TemporaryDirectory(prefix="credit_xl_test_") as d:
        yield Path(d)


@pytest.fixture(scope="module")
def report(tmp_dir):
    return rx.build_and_reconcile(quick=True, xlsx_out=tmp_dir / "wb.xlsx", report_out=tmp_dir / "report.json")


@pytest.fixture(scope="module")
def paths(report, tmp_dir):
    return tmp_dir / "wb.xlsx", tmp_dir / "wb.meta.json"


@pytest.fixture(scope="module")
def wb(paths):
    return openpyxl.load_workbook(paths[0])


@pytest.fixture(scope="module")
def meta(paths):
    return json.loads(paths[1].read_text())


@needs_lo
def test_reconciliation_passes(report, tmp_dir):
    assert report["all_passed"], [c["issues"] for c in report["cases"] if not c["passed"]]
    assert report["n_cases"] >= MIN_CASES
    assert json.loads((tmp_dir / "report.json").read_text())["data_source"]


@needs_lo
def test_plain_formatting(wb):
    for ws in wb:
        assert ws.sheet_properties.tabColor is None, ws.title
        for row in ws.iter_rows():
            for c in row:
                if c.value is None:
                    continue
                assert c.fill is None or c.fill.fill_type in (None, "none"), (ws.title, c.coordinate)
                color = c.font.color
                assert color is None or color.type != "rgb" or color.rgb in ("FF000000", "00000000"), (ws.title, c.coordinate)


@needs_lo
def test_author_metadata(wb, paths):
    assert wb.properties.creator == rx.AUTHOR
    assert wb.properties.lastModifiedBy == rx.AUTHOR
    with zipfile.ZipFile(paths[0]) as z:
        assert rx.AUTHOR in z.read("docProps/core.xml").decode("utf-8")


@needs_lo
def test_formula_sheets_have_no_typed_numbers(wb):
    for name in FORMULA_SHEETS:
        for row in wb[name].iter_rows():
            for c in row:
                v = c.value
                assert not (isinstance(v, (int, float)) and not isinstance(v, bool)), (name, c.coordinate, v)


@needs_lo
def test_mixed_sheets_type_only_reference_columns(wb, meta):
    for name in MIXED_SHEETS:
        allowed = set(meta["typed_cols"][name])
        for row in wb[name].iter_rows(min_row=4):
            for c in row:
                v = c.value
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    assert c.column_letter in allowed, (name, c.coordinate, v)


@needs_lo
def test_loan_sample_live_columns_are_formulas(wb, meta):
    ws = wb["Loan_Sample"]
    for h, letter in meta["ls_cols"].items():
        if not h.startswith(LIVE_PREFIX):
            continue
        for r in (meta["first"], (meta["first"] + meta["last"]) // 2, meta["last"]):
            v = ws[f"{letter}{r}"].value
            assert isinstance(v, str) and v.startswith("="), (h, r, v)


@needs_lo
def test_conclusions_static(wb):
    for name in STATIC_SHEETS:
        for row in wb[name].iter_rows():
            for c in row:
                assert not (isinstance(c.value, str) and c.value.startswith("=")), (name, c.coordinate)


@needs_lo
def test_no_dashes_in_workbook(wb):
    for ws in wb:
        for row in ws.iter_rows():
            for c in row:
                if isinstance(c.value, str):
                    assert chr(0x2014) not in c.value and chr(0x2013) not in c.value, (ws.title, c.coordinate)


@needs_lo
def test_checks_cell_reports_pass(paths, meta):
    wbv = openpyxl.load_workbook(paths[0], data_only=True)
    sheet, ref = meta["addr"]["checks_all"].split("!")
    assert wbv[sheet][ref].value == 1


@needs_lo
def test_bin_rule_matches_searchsorted(tmp_dir):
    """Excel bin lookup on boundary values equals numpy searchsorted(side="left") + 1."""
    sc = rx.load_scorecard()
    spec = sc.specs[next(v for v in sc.variables if sc.specs[v].kind == "numeric")]
    edges = np.asarray(spec.edges, dtype=np.float64)
    probes = np.unique(np.r_[edges, edges + REL_STEP * np.maximum(np.abs(edges), 1.0), edges - REL_STEP * np.maximum(np.abs(edges), 1.0), EDGE_PROBES])
    expected = np.searchsorted(edges, probes, side="left") + 1
    wbk = openpyxl.Workbook()
    ws = wbk.active
    for i, e in enumerate(edges, start=1):
        ws.cell(i, 1, float(e))
    ecol = f"$A$1:$A${len(edges)}"
    for j, x in enumerate(probes, start=1):
        ws.cell(j, 3, float(x))
        ws.cell(j, 4, f"=SUMPRODUCT(--({ecol}<C{j}))+1")
        ws.cell(j, 5, f'=COUNTIF({ecol},"<"&C{j})+1')
    src = tmp_dir / "binrule.xlsx"
    wbk.save(src)
    out = rx.recalc(src, tmp_dir / "binrule_out", rx.find_soffice())
    v = openpyxl.load_workbook(out, data_only=True).active
    sp = np.array([v.cell(j, 4).value for j in range(1, len(probes) + 1)])
    ci = np.array([v.cell(j, 5).value for j in range(1, len(probes) + 1)])
    assert (sp == expected).all()
    # COUNTIF with text criteria is allowed to differ only for values that sit within 15 digits of an edge
    bad = probes[ci != expected]
    assert all(np.min(np.abs(edges - b)) < 1e-9 * max(1.0, abs(b)) for b in bad)
