"""The private-overlay spec discovery hook (Workstream C/H keystone)."""

from __future__ import annotations

import textwrap

from modelfactory.datasets import specs


def test_overlay_specs_register(tmp_path, monkeypatch):
    overlay = tmp_path / "specs"
    overlay.mkdir()
    (overlay / "extra.py").write_text(
        textwrap.dedent(
            """
            from modelfactory.datasets.specs import _register, DatasetSpec, StructureMapping
            _register(
                DatasetSpec(
                    dataset_id=987,
                    name="OverlayTest",
                    description="overlay discovery probe",
                    structures=(StructureMapping(canonical="Foo"),),
                    tags={"region": "test"},
                ),
                key="overlay_test_987",
            )
            """
        )
    )
    # A leading-underscore module is a shared helper and must be skipped.
    (overlay / "_helpers.py").write_text("raise RuntimeError('should not be imported')\n")

    monkeypatch.setenv("MFACTORY_SPECS_OVERLAY", str(overlay))
    specs._load_overlay_specs()

    assert "overlay_test_987" in specs.SPECS
    assert specs.SPECS["overlay_test_987"].dataset_id == 987
    assert specs.SPECS["overlay_test_987"].folder == "Dataset987_OverlayTest"


def test_no_overlay_is_a_noop(monkeypatch):
    # Pointing at a non-existent dir must not raise.
    monkeypatch.setenv("MFACTORY_SPECS_OVERLAY", "/nonexistent/overlay/dir")
    specs._load_overlay_specs()
