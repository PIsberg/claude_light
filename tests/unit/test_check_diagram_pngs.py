"""
Tests for tests/linting/check_diagram_pngs.py, the PR check that a changed
PlantUML source comes with its re-rendered PNG (#107). indexing_flow.puml
was edited while Claude_Light_Indexing_Flow.png stayed months older (#67).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tests.linting.check_diagram_pngs import png_for, stale_diagrams

ROOT = Path(__file__).resolve().parent.parent.parent


class TestStaleDiagrams(unittest.TestCase):

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="cl_puml_"))
        d = self.root / "docs" / "plantuml"
        d.mkdir(parents=True)
        (d / "flow.puml").write_text("@startuml Flow_Diagram\n:a;\n@enduml\n", encoding="utf-8")

    def test_source_changed_without_its_png_is_stale(self):
        self.assertEqual(stale_diagrams(["docs/plantuml/flow.puml"], self.root),
                         [("docs/plantuml/flow.puml", "docs/plantuml/Flow_Diagram.png")])

    def test_source_changed_with_its_png_passes(self):
        self.assertEqual(stale_diagrams(["docs/plantuml/flow.puml",
                                         "docs/plantuml/Flow_Diagram.png"], self.root), [])

    def test_other_changes_are_ignored(self):
        self.assertEqual(stale_diagrams(["README.md", "docs/plantuml/Flow_Diagram.png"], self.root), [])

    def test_deleted_source_is_ignored(self):
        self.assertEqual(stale_diagrams(["docs/plantuml/gone.puml"], self.root), [])

    def test_png_name_comes_from_startuml(self):
        # The PNG is named by @startuml, not by the file name.
        self.assertEqual(png_for(self.root / "docs/plantuml/flow.puml"), "Flow_Diagram.png")

    def test_every_diagram_in_the_repo_has_its_png(self):
        # The check maps through @startuml; a source whose PNG has another
        # name would always look stale.
        for puml in (ROOT / "docs" / "plantuml").glob("*.puml"):
            self.assertTrue((puml.parent / png_for(puml)).is_file(), puml.name)


if __name__ == "__main__":
    unittest.main()
