"""
Placeholder test to bootstrap CI.
Will be replaced/expanded as real modules (teacher, student, distillation) are added.
"""


def test_placeholder():
    """Sanity check that pytest and the CI pipeline are wired up correctly."""
    assert 1 + 1 == 2


def test_project_importable():
    """Sanity check that the src package is importable once installed/on path."""
    import src  # noqa: F401
    