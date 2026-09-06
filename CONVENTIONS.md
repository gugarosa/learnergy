# Learnergy conventions

Learnergy adopts the applicable code-style rules from cpmux's phitrain-derived conventions.
These rules govern new code and convention updates without authorizing unrelated algorithm or API changes.

## Compatibility and scope

- Keep the current Python 3.11 minimum. Use the requested modern Python idioms where they are compatible with it.
- Preserve public imports, model names, constructor options, return values, normalization policies, and checkpoint state.
- Keep Learnergy's domain-oriented package layout and public package exports. cpmux's empty package initializers,
  CLI architecture, Pydantic choices, and test-layout migration are not part of this adoption.
- Tests remain grouped by the existing model families. Runtime validation rules do not prohibit test assertions.

## Code style

- Use `X | None`, builtin generics such as `list[str]`, and ABCs from `collections.abc`.
  Import only typing-specific constructs such as `Any` and `Literal` from `typing`. (R2)
- Start every Python file with the project header:

  ```python
  # Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
  # Licensed under the Apache License, Version 2.0.
  ```

- Keep imports top-level, absolute, and grouped as standard library, third-party, then Learnergy imports.
- Use double-quoted strings. Code formatting and readable prose use a 120-character limit. (R9)
- Public functions, regular classes, and explicit constructors have Google-style docstrings.
  Use a single-sentence summary and one-line `Args:`, `Returns:`, and `Raises:` entries as applicable.
  Do not put semicolons or `defaults to ...` tails in entries. (R3, R13)
- A regular class has a one-sentence class summary. Constructor arguments belong on `__init__`, not the class.
  Preserve mathematical references and substantive behavioral notes in module or operation documentation.
- Multiline docstrings have one blank line before the closing triple quotes and one blank line after them
  before code or fields. Keep class and module summaries on one line when they need no other content,
  matching cpmux's examples and Black's normalization.
- Private helpers have no docstrings. Framework-dispatched overrides such as `forward`, `__getitem__`, and
  `__len__` have no docstrings. Document their relevant contracts in public sampling methods, constructors,
  module documentation, or the user guide instead.
- Data classes without an explicit constructor document each field on one line in an `Attributes:` section.
  This rule does not require introducing data classes into existing models.
- Public contracts describe tensor shapes, tuple element order, mutation, history, normalization, gradient
  boundaries, and failure behavior where relevant. An annotation is not runtime validation.
- Obtain library loggers with `get_logger(__name__)` from `learnergy.utils.logging`.
  Do not use `print()` in library code. Application examples may print their results.
- Warning and error diagnostics identify a backticked offender and end with a period, for example
  `` f"`name={value}` could not be loaded." ``. Info and debug messages stay plain. (R14)
- Raised messages use `` "`name` <verb phrase>[, but got <value>]." ``.
  Use `is None` and `is True` prose rather than comparison operators in messages. (R1)
- Validate with `if` and a specific raised exception, never a runtime `assert`. Do not use bare `except:`.
- Comments explain why rather than narrating the next statement. Prefer no comment or one line, with a
  three-line maximum, no banners, and no trailing period. Copyright and license notices are exempt. (R8)
- Separate logical phases with one blank line in function bodies of at least 12 lines. (R11)
- Inline new single-use implementation details. Extract a helper, constant, or configurable parameter when
  a second call-site establishes reuse. Do not delete published APIs or required callbacks by counting only
  their internal call-sites. (R16)

## Tests and review

- Test functions and fixtures are plain functions without docstrings. Test functions have no type annotations.
- Use bare assertions without failure-message strings. Names should describe the observable behavior.
- Preserve seeded behavior when claiming a behavior-preserving refactor. Treat deliberate numerical changes
  as separate, explicit changes rather than hiding them in a style pass.
- Regression tests compare actual model output with an independently justified expectation and should fail
  for the defect they claim to prevent.

## Tooling

Use the existing pinned Black, isort, and flake8 tools at line length 120.
The supported interpreter range, dependencies, and build backend do not change as part of this style adoption.

```text
uv sync --locked --extra docs
uv run pre-commit run --all-files
uv run pytest
uv run python -m sphinx -W -b html docs build
```
