"""
Tests for ``import_tracker.track_imports``.

These tests pin the semantics of the ``track_imports`` context manager using
only the public package API (``import import_tracker``):

1. **Nesting**: when tracking is entered nested inside another active scope,
   each scope keeps its own record. An import inside the inner scope is seen by
   both scopes; after the inner scope exits, the outer scope keeps recording.
2. **Repeated imports**: importing the same module several times in one scope
   records it exactly once.
3. **Parent/child capture**: both ``from pkg import sub`` and
   ``import pkg.sub`` capture the parent package together with the child
   module, in a reproducible order (parent before child).
4. **Re-entrancy / cleanup**: a fixture snapshots and restores
   ``sys.modules`` (plus the rest of the import machinery); an exception in
   the ``with`` body propagates unchanged while the import hooks are restored
   to their exact entry state, so later tests never see earlier state.

Implementation status
---------------------
In this checkout the package exports only ``track_module``, ``setup_tools``
and ``lazy_import_errors``; there is no ``track_imports`` symbol in the
package or upstream. The semantic tests therefore exercise a genuine
implementation gap rather than a test mistake, so they are marked
``xfail(strict=True)`` with the reason spelled out in each docstring. If the
feature ever lands with different semantics, ``strict=True`` makes those
tests fail loudly; once it lands with these semantics they pass normally.

The HTML artifact test does not depend on the missing feature and always
runs, so ``test/artifacts/import_trace.html`` is produced for the screen
recording regardless of implementation status.
"""

# Standard
from contextlib import contextmanager
import builtins
import html
import importlib
import os
import sys
import textwrap

# Third Party
import pytest

# Local
import import_tracker

ARTIFACTS_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "artifacts"))
ARTIFACT_HTML = os.path.join(ARTIFACTS_DIR, "import_trace.html")


## Helpers ####################################################################


def _get_track_imports():
    """Return the public ``track_imports`` callable.

    Raising here (instead of ``pytest.skip``) is what makes the surrounding
    ``xfail`` report the missing API as an expected failure.
    """
    try:
        track_imports = import_tracker.track_imports
    except AttributeError:
        raise AssertionError(
            "import_tracker does not expose a public 'track_imports' API"
        )
    if not callable(track_imports):
        raise AssertionError("import_tracker.track_imports is not callable")
    return track_imports


def _tracked_names(tracker):
    """Extract an ordered, de-duplicated list of names from ``tracker``.

    The tracker is the value bound by ``as`` in
    ``with track_imports() as tracker``. The expected public surface is an
    ordered view of captured module names; accept a few sensible attribute
    names or an iterable tracker while preserving order.
    """
    for attr in ("imports", "modules", "names"):
        value = getattr(tracker, attr, None)
        if value is not None:
            return list(value)
    return list(tracker)


@contextmanager
def _frozen_import_state():
    """Snapshot and fully restore the process import machinery."""
    saved_modules = dict(sys.modules)
    saved_meta_path = list(sys.meta_path)
    saved_path_hooks = list(sys.path_hooks)
    saved_path_importer_cache = dict(sys.path_importer_cache)
    saved_path = list(sys.path)
    saved_import = builtins.__import__
    try:
        yield
    finally:
        builtins.__import__ = saved_import
        sys.path[:] = saved_path
        sys.meta_path[:] = saved_meta_path
        sys.path_hooks[:] = saved_path_hooks
        sys.path_importer_cache.clear()
        sys.path_importer_cache.update(saved_path_importer_cache)
        sys.modules.clear()
        sys.modules.update(saved_modules)


def _write_file(path, contents):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(textwrap.dedent(contents).lstrip())


@pytest.fixture
def clean_import_state():
    """Give every test a pristine import environment and restore it after."""
    with _frozen_import_state():
        yield


@pytest.fixture
def demo_site(tmp_path, clean_import_state):
    """Create tiny packages guaranteed not to be pre-imported."""
    base = str(tmp_path)
    for pkg_name, child_name, value in [
        ("ti_demo_alpha", "child", 1),
        ("ti_demo_beta", "child", 2),
        ("ti_demo_outer", "mod", "True"),
        ("ti_demo_inner", "mod", "True"),
    ]:
        pkg_dir = os.path.join(base, pkg_name)
        os.makedirs(pkg_dir, exist_ok=True)
        _write_file(
            os.path.join(pkg_dir, "__init__.py"),
            '"""demo package for track_imports tests"""\n',
        )
        _write_file(os.path.join(pkg_dir, child_name + ".py"), "VALUE = %s\n" % value)
    sys.path.insert(0, base)
    return base


def _tree_from_ordered_names(names):
    """Build a nested dict tree from parent-before-child dotted names."""
    tree = {}
    for name in names:
        node = tree
        for part in name.split("."):
            node = node.setdefault(part, {})
    return tree


def _render_tree_html(tree, depth=0):
    """Render a nested module-name dict as nested HTML lists."""
    lines = []
    for key in sorted(tree):
        lines.append(
            '<li class="module depth%d">%s</li>' % (depth, html.escape(key, quote=True))
        )
        if tree[key]:
            lines.append("<ul>")
            lines.append(_render_tree_html(tree[key], depth + 1))
            lines.append("</ul>")
    return "\n".join(lines)


def _write_artifact(sections):
    """Write captured import trees to the screen-recording HTML artifact.

    ``sections`` is an ordered list of (heading, ordered-names) pairs.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    body_parts = []
    for heading, names in sections:
        tree = _tree_from_ordered_names(names)
        body_parts.append(
            "<section><h2>%s</h2><ul>%s</ul></section>"
            % (html.escape(heading), _render_tree_html(tree))
        )
    document = (
        textwrap.dedent(
            """
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8">
        <title>import_tracker track_imports trace</title>
        <style>
        body { font-family: sans-serif; margin: 2em; }
        h1 { font-size: 1.4em; }
        h2 { font-size: 1.1em; margin-top: 1.5em; }
        ul { list-style: none; padding-left: 1.2em;
             border-left: 1px solid #bbb; }
        li.module { font-family: monospace; }
        .depth0 { font-weight: bold; }
        section { margin-bottom: 1em; }
        </style>
        </head>
        <body>
        <h1>Captured import tree</h1>
        %s
        </body>
        </html>
        """
        ).strip()
        % "\n".join(body_parts)
    )
    with open(ARTIFACT_HTML, "w", encoding="utf-8") as handle:
        handle.write(document)


## Public API surface #########################################################


@pytest.mark.xfail(
    strict=True,
    reason=(
        "track_imports is not implemented: the public package namespace has "
        "no such callable"
    ),
)
def test_track_imports_is_public_callable():
    """``track_imports`` must be reachable from the public package namespace.

    xfail reason: this checkout has no such symbol; only ``track_module``,
    ``setup_tools`` and ``lazy_import_errors`` are exported.
    """
    assert callable(_get_track_imports())


## Semantics ##################################################################


@pytest.mark.xfail(
    strict=True,
    reason=(
        "track_imports is not implemented: no context manager exists to "
        "provide independent nested capture scopes"
    ),
)
def test_nested_scopes_record_independently_and_outer_resumes(demo_site):
    """Nested scopes each keep a copy; the outer resumes after the inner exits.

    An import made inside the inner scope is visible to both the inner and
    outer trackers. After the inner scope closes, a later import is recorded
    by the outer tracker and must not leak into the closed inner record.

    xfail reason: there is no track_imports context manager in this checkout,
    so the nested-scope behavior cannot exist yet.
    """
    track_imports = _get_track_imports()

    with track_imports() as outer:
        importlib.import_module("ti_demo_outer")
        with track_imports() as inner:
            importlib.import_module("ti_demo_inner")
            inner_names = _tracked_names(inner)
        outer_names_mid = _tracked_names(outer)
        importlib.import_module("ti_demo_outer.mod")
        outer_names_end = _tracked_names(outer)

    assert "ti_demo_inner" in inner_names
    assert "ti_demo_outer" not in inner_names
    assert "ti_demo_outer" in outer_names_mid
    assert "ti_demo_inner" in outer_names_mid
    assert "ti_demo_outer.mod" in outer_names_end


@pytest.mark.xfail(
    strict=True,
    reason=(
        "track_imports is not implemented: de-duplicated recording cannot be "
        "exercised without the context manager"
    ),
)
def test_repeated_import_is_deduplicated(demo_site):
    """The same module imported repeatedly is recorded exactly once.

    Design choice, locked here deliberately: **de-duplicate**. A tracker
    answers the question "which modules did this block depend on?", so every
    repeat of an already captured name collapses to one entry. Recording each
    statement would instead make the output depend on incidental import style
    and on the ``sys.modules`` cache state, and it would break reproducible
    ordering. The single retained entry keeps its first-seen position, which
    stays deterministic.

    xfail reason: track_imports does not exist in this checkout.
    """
    track_imports = _get_track_imports()

    with track_imports() as tracker:
        importlib.import_module("ti_demo_alpha")
        importlib.import_module("ti_demo_alpha")
        importlib.import_module("ti_demo_alpha.child")
        importlib.import_module("ti_demo_alpha.child")
        names = _tracked_names(tracker)

    assert names.count("ti_demo_alpha") == 1
    assert names.count("ti_demo_alpha.child") == 1


@pytest.mark.xfail(
    strict=True,
    reason=(
        "track_imports is not implemented: parent/child capture order cannot "
        "be verified without the context manager"
    ),
)
def test_from_and_dotted_import_capture_parent_and_child_in_order(demo_site):
    """Both import forms capture parent + child in a reproducible order.

    ``from pkg import child`` and ``import pkg.child`` both execute the parent
    package before binding the submodule, so each form must record both names
    with the parent preceding the child. Two fresh packages keep the forms
    independent of each other's ``sys.modules`` cache, and repeating a cached
    import must still report the same parent-before-child order.

    xfail reason: track_imports does not exist in this checkout.
    """
    track_imports = _get_track_imports()

    with track_imports() as from_tracker:
        exec("from ti_demo_alpha import child", {"__name__": __name__})
        from_names = _tracked_names(from_tracker)

    with track_imports() as dotted_tracker:
        exec("import ti_demo_beta.child", {"__name__": __name__})
        dotted_names = _tracked_names(dotted_tracker)

    for names, parent, child in [
        (from_names, "ti_demo_alpha", "ti_demo_alpha.child"),
        (dotted_names, "ti_demo_beta", "ti_demo_beta.child"),
    ]:
        assert parent in names
        assert child in names
        assert names.index(parent) < names.index(child)

    # A repeat import is served from the sys.modules cache; ordering must stay
    # reproducible and de-duplicated.
    with track_imports() as again:
        exec("import ti_demo_beta.child", {"__name__": __name__})
        again_names = _tracked_names(again)

    assert again_names.index("ti_demo_beta") < again_names.index("ti_demo_beta.child")
    assert again_names.count("ti_demo_beta.child") == 1


@pytest.mark.xfail(
    strict=True,
    reason=(
        "track_imports is not implemented: context-manager cleanup on the "
        "exception path cannot be verified without the context manager"
    ),
)
def test_exception_propagates_and_hooks_restore(demo_site):
    """An exception in the body still raises, and import hooks are restored.

    On the exception path the tracker must behave like a well-formed context
    manager: the original exception propagates unchanged, ``builtins.__import__``
    is the very same object as on entry, and nothing extra is left installed
    on ``sys.meta_path`` / ``sys.path_hooks``.

    xfail reason: track_imports does not exist in this checkout.
    """
    track_imports = _get_track_imports()

    before_import = builtins.__import__
    before_meta_path = list(sys.meta_path)
    before_path_hooks = list(sys.path_hooks)

    class Boom(RuntimeError):
        pass

    with pytest.raises(Boom):
        with track_imports():
            importlib.import_module("ti_demo_alpha")
            raise Boom("body failure")

    assert builtins.__import__ is before_import
    assert sys.meta_path == before_meta_path
    assert sys.path_hooks == before_path_hooks


@pytest.mark.xfail(
    strict=True,
    reason=(
        "track_imports is not implemented: normal-exit hook cleanup cannot be "
        "verified without the context manager"
    ),
)
def test_hooks_restore_on_normal_exit(demo_site):
    """Leaving a scope normally leaves no installed import hook behind."""
    track_imports = _get_track_imports()

    before_import = builtins.__import__
    before_meta_path = list(sys.meta_path)

    with track_imports():
        importlib.import_module("ti_demo_alpha")

    assert builtins.__import__ is before_import
    assert sys.meta_path == before_meta_path


## Re-entrancy across tests ###################################################


def test_state_is_clean_before_consumer(demo_site):
    """The demo packages must not already be present in ``sys.modules``."""
    assert "ti_demo_alpha" not in sys.modules
    assert "ti_demo_outer.mod" not in sys.modules


def test_following_test_sees_no_pollution(demo_site):
    """Demo packages imported by earlier tests are absent for a later test."""
    assert "ti_demo_alpha" not in sys.modules
    assert "ti_demo_beta.child" not in sys.modules
    assert "ti_demo_inner.mod" not in sys.modules


## HTML artifact for screen recording #########################################


def test_renders_import_trace_html_artifact(demo_site):
    """Render the captured import tree with the stdlib ``html`` module.

    Use ``track_imports`` when available so the artifact shows the real
    feature; until then capture names from the ``sys.modules`` delta (public
    behavior) so the HTML is always produced. Only the demo names are rendered
    so the artifact is stable across environments, and it is written to
    ``test/artifacts/import_trace.html``.
    """
    captured = {}
    track_imports = getattr(import_tracker, "track_imports", None)

    def capture(label, import_callable):
        if track_imports is not None:
            with track_imports() as tracker:
                import_callable()
            captured[label] = _tracked_names(tracker)
            return
        before = set(sys.modules)
        import_callable()
        captured[label] = sorted(set(sys.modules) - before)

    capture(
        "from ti_demo_alpha import child",
        lambda: exec("from ti_demo_alpha import child", {"__name__": __name__}),
    )
    capture(
        "import ti_demo_beta.child",
        lambda: exec("import ti_demo_beta.child", {"__name__": __name__}),
    )

    # Render only the demo names so the artifact is stable across environments.
    demo_sections = []
    for label, names in captured.items():
        demo_names = [
            name
            for name in names
            if name == "ti_demo_alpha"
            or name.startswith("ti_demo_alpha.")
            or name == "ti_demo_beta"
            or name.startswith("ti_demo_beta.")
        ]
        demo_sections.append((label, demo_names))

    _write_artifact(demo_sections)

    assert os.path.isfile(ARTIFACT_HTML)
    with open(ARTIFACT_HTML, encoding="utf-8") as handle:
        content = handle.read()
    assert "ti_demo_alpha" in content
    assert "ti_demo_beta.child" in content
    # Every rendered string went through html escaping.
    assert "<script>" not in content.lower()
