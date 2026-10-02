"""
Tests for ``import_tracker.track_imports``.

These tests pin down the semantics of the ``track_imports`` context manager
through the public package API only (``import import_tracker``):

1. **Nesting**: entering ``track_imports`` nested inside another active
   tracking scope keeps an independent record for each scope; imports made in
   the inner scope are visible to both scopes, and the outer scope keeps
   recording after the inner scope exits.
2. **Repeated imports**: the same module imported more than once in a scope is
   recorded once (de-duplicated), not once per statement.
3. **Parent/child capture**: both ``from pkg import sub`` and
   ``import pkg.sub`` capture the parent package *and* the child module, in a
   deterministic, reproducible order.
4. **Cleanup / re-entrancy**: leaving the scope (normally or via an exception)
   restores the process import machinery to exactly what it was on entry, and
   the exception from the ``with`` body still propagates.

Implementation status
---------------------
As of this checkout ``import_tracker`` only exports ``track_module``,
``setup_tools`` and ``lazy_import_errors``; there is no ``track_imports``
symbol anywhere in the package or upstream. The semantic tests below therefore
fail today for a genuine implementation gap, not for a test bug, so they are
marked ``xfail(strict=True)`` with each docstring recording why. They become
loud failures (via ``strict=True``) if the feature lands with different
semantics, and expected passes once it lands with these semantics.

The HTML rendering test does not depend on the missing feature and always
runs, so that ``test/artifacts/import_trace.html`` is produced for screen
recording regardless of the feature's status.
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

    Raising here (rather than ``pytest.skip``) is what makes the surrounding
    ``xfail`` mark report the missing API as the expected failure.
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
    """Extract the ordered, de-duplicated fully-qualified names from tracker.

    The tracker is the value bound by ``as`` in
    ``with track_imports() as tracker``. The expected public surface is an
    ordered view of captured module names; support a few reasonable attribute
    names or a directly iterable tracker while preserving order.
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
        _write_file(
            os.path.join(pkg_dir, child_name + ".py"),
            "VALUE = %s\n" % value,
        )

    sys.path.insert(0, base)
    return base


def _tree_from_ordered_names(names):
    """Build a nested dict tree from parent-before-child dotted names."""
    tree = {}
    for name in names:
        parts = name.split(".")
        node = tree
        for part in parts:
            node = node.setdefault(part, {})
    return tree


def _render_tree_html(tree, depth=0):
    """Render a nested module-name dict as nested HTML <ul> lists."""
    lines = []
    for key in sorted(tree):
        classes = "module depth%d" % depth
        lines.append('<li class="%s">%s</li>' % (classes, html.escape(key, quote=True)))
        if tree[key]:
            lines.append("<ul>")
            lines.append(_render_tree_html(tree[key], depth + 1))
            lines.append("</ul>")
    return "\n".join(lines)


def _write_artifact(sections):
    """Write the captured import trees to the screen-recording HTML artifact.

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

    xfail reason: the symbol does not exist in this checkout; only
    ``track_module`` / ``setup_tools`` / ``lazy_import_errors`` are exported.
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
    """Nested scopes each keep a copy; the outer keeps accumulating after.

    An import made inside the inner scope is visible to both the inner and
    outer trackers. After the inner scope exits, a later import is recorded by
    the outer tracker but must not leak back into the closed inner record.

    xfail reason: there is currently no track_imports context manager, so the
    nested-scope behavior cannot exist yet.
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

    Design choice locked here on purpose: de-duplicate. A tracker answers
    "which modules did this block depend on?", so every repeat of an already
    captured name collapses to a single entry. Recording each statement would
    instead make the output depend on incidental import style and on the
    sys.modules cache state, and would break reproducible ordering. The single
    retained entry keeps its first-seen position, which stays deterministic.

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
    """Both import forms capture parent + child, reproducibly ordered.

    ``from pkg import child`` and ``import pkg.child`` both execute the parent
    package before binding the submodule, so each form must record both names
    with the parent preceding the child. Two fresh packages keep the forms
    independent of one another's sys.modules cache.

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

    with track_imports() as again:
        exec("import ti_demo_beta.child", {"__name__": __name__})
        again_names = _tracked_names(again)
        assert again_names.index("ti_demo_beta") < again_names.index(
            "ti_demo_beta.child"
        )


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
    is the very same object as on entry, and nothing was left installed on
    ``sys.meta_path`` / ``sys.path_hooks``.

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
    """Paired check: demo packages must not already be in sys.modules."""
    assert "ti_demo_alpha" not in sys.modules
    assert "ti_demo_outer.mod" not in sys.modules


def test_following_test_sees_no_pollution(demo_site):
    """After earlier tests imported demo packages, they are absent again."""
    assert "ti_demo_alpha" not in sys.modules
    assert "ti_demo_beta.child" not in sys.modules
    assert "ti_demo_inner.mod" not in sys.modules


## HTML artifact for screen recording #########################################


def test_renders_import_trace_html_artifact(demo_site):
    """Render the captured import tree with the stdlib ``html`` module.

    Prefer ``track_imports`` when available so the artifact shows the real
    feature; until then capture names from the sys.modules delta (public
    behavior) so the HTML is always produced. Output ordering is made
    reproducible, and the file is written to test/artifacts/import_trace.html.
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

    # Only render the demo names so the artifact is stable across environments
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
    # All rendered text passed through html escaping
    assert "<script>" not in content.lower()
