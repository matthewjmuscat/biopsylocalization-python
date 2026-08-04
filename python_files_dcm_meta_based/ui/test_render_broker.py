"""Synthetic checks for render broker contracts."""

from __future__ import annotations

import unittest

from ui.render_broker import RenderBrokerChoiceGroup
from ui.render_broker import RenderBrokerRequest
from ui.render_broker import normalize_render_backend
from ui.render_broker import normalize_render_broker_request
from ui.render_broker import render_backend_includes
from ui.render_broker import resolve_render_backend


class RenderBrokerTests(unittest.TestCase):
    def test_normalize_render_backend_preserves_legacy_both_alias(self) -> None:
        self.assertEqual(normalize_render_backend("both"), "both")
        self.assertEqual(normalize_render_backend("open3d+plotly"), "both")
        self.assertTrue(render_backend_includes("both", "open3d"))
        self.assertTrue(render_backend_includes("both", "plotly"))
        self.assertFalse(render_backend_includes("both", "pyvista"))

    def test_normalize_render_backend_accepts_pyvista_combinations(self) -> None:
        self.assertEqual(normalize_render_backend("pyvista"), "pyvista")
        self.assertEqual(normalize_render_backend("plotly,pyvista"), "plotly+pyvista")
        self.assertEqual(normalize_render_backend("pyvista+open3d"), "open3d+pyvista")
        self.assertTrue(render_backend_includes("open3d+pyvista", "pyvista"))

    def test_resolve_render_backend_accepts_pyvista_flag(self) -> None:
        self.assertEqual(resolve_render_backend(False, False, True), "pyvista")
        self.assertEqual(resolve_render_backend(True, True, False), "both")
        self.assertEqual(resolve_render_backend(True, False, True), "open3d+pyvista")

    def test_normalize_request_preserves_pyvista_choice_group(self) -> None:
        request = RenderBrokerRequest(
            title="test",
            choice_groups=(
                RenderBrokerChoiceGroup(
                    group_key="group",
                    display_label="Group",
                    allow_pyvista=True,
                    default_backend="pyvista",
                ),
            ),
        )

        normalized_request = normalize_render_broker_request(request)

        self.assertTrue(normalized_request.choice_groups[0].allow_pyvista)
        self.assertEqual(normalized_request.choice_groups[0].default_backend, "pyvista")


if __name__ == "__main__":
    unittest.main()