# Private Scenarios and Attack Techniques

PyRIT plug-ins let an operator activate private scenarios and attack techniques from a
**stock PyRIT installation** — without forking PyRIT or passing per-run CLI flags.

## Plug-in or `--initialization-scripts`?

A plug-in is a thin layer over the existing initializer path:

- `--initialization-scripts ./my_init.py` runs a custom `PyRITInitializer` from a loose
  file. It works for a self-contained script but is per-run and breaks for a packaged
  initializer that imports from its own package.
- A `plugins:` entry in `.pyrit_conf` names a private initializer once, runs it
  **first** (before other initializers and before catalog warming), and anchors the
  package root on `sys.path` so a packaged initializer with intra-package imports loads
  correctly.

Use a plug-in when a private, packaged initializer (for example a team's internal
red-teaming package) needs to behave like a built-in inside the standard backend, GUI,
and `pyrit_scan` catalog.

## What a plug-in is

A plug-in is a config entry that points at a concrete `PyRITInitializer` reachable by a
dotted path from a source root:

```yaml
# .pyrit_conf
plugins:
  - name: my_redteam
    source: /repos/my-redteam
    initializer: my_redteam.setup.MyInitializer
```

- `name` — an operator label used in logs and errors.
- `source` — the folder placed on `sys.path` so your package can be imported (see below).
- `initializer` — a dotted `module.Class` path to a concrete `PyRITInitializer`.

### What `source` should point at (and why it matters)

`source` should be the folder that **contains** your package — the directory you would be
sitting in for `import my_redteam` to succeed at a Python prompt. PyRIT adds that folder
to Python's import search path (`sys.path`) before importing your initializer.

You need this because a real private package doesn't live next to PyRIT; its modules
import from *each other* (for example `from my_redteam.datasets import load`). If Python
can't find the package root, those imports fail and the plug-in won't load. In plain
terms: point `source` at the folder above your package, not at the package folder itself
and not at a single file buried inside it. If you point at the wrong place, loading fails
closed with an import error naming what could not be found.

`ConfigurationLoader` runs the plug-in as a privileged initializer, always **first** —
before your other initializers and before anything reads the scenario/technique catalog.
You do not (and cannot) add it to `initializers:`: it isn't a registered initializer name,
so listing it there fails with an "initializer not found" error. The framework constructs
and runs it for you.

## The initializer owns registration

PyRIT discovers nothing on its own. The plug-in's initializer registers everything it
wants discoverable, at whatever level of abstraction fits:

- **Attack techniques** —
  `AttackTechniqueRegistry.get_registry_singleton().register_from_factories([...])`;
  selectable via `--techniques`.
- **Scenarios** —
  `ScenarioRegistry.get_registry_singleton().register_class(MyScenario, name="my_redteam.violence")`;
  runnable via `pyrit_scan my_redteam.violence`.
- **Datasets** — register providers and load them into memory so private seeds stay in
  the operator's database and are never published.
- **Default targets** — `set_default_value(...)`.

### Deciding what to keep private

"Private" is rarely all-or-nothing. If you build a custom scenario or technique, you have
to decide whether to contribute it publicly, keep it in your own tracked repo, or keep it
fully private — and often only *part* of it is sensitive. A plug-in lets you keep exactly
the sensitive layer private while everything else stays public and works out of the box:

- sometimes only the **dataset** (the prompts/objectives) is sensitive;
- sometimes it's a niche **technique**;
- sometimes an entire **scenario** should not be exposed at all.

Because the initializer registers each level independently, you can publish the generic
parts and register only the sensitive parts from your private package.

### Example initializer

```python
from pyrit.executor.attack import PromptSendingAttack
from pyrit.registry import AttackTechniqueRegistry, ScenarioRegistry
from pyrit.scenario.core import AttackTechniqueFactory
from pyrit.setup.pyrit_initializer import PyRITInitializer

from my_redteam.scenarios import Violence


class MyInitializer(PyRITInitializer):
    """Register a private technique and a private scenario."""

    async def initialize_async(self) -> None:
        AttackTechniqueRegistry.get_registry_singleton().register_from_factories(
            [AttackTechniqueFactory(name="operation_foobar", attack_class=PromptSendingAttack)]
        )
        ScenarioRegistry.get_registry_singleton().register_class(Violence, name="my_redteam.violence")
```

## Usage

```powershell
# A private technique through a public scenario
pyrit_scan airt.rapid_response --target openai_chat --techniques operation_foobar

# A private scenario
pyrit_scan my_redteam.violence --target openai_chat
```

## Behavior and limits

- The plug-in initializer runs **first**, so lazy catalog/metadata consumers see a
  complete registry.
- Loading executes third-party Python with backend permissions; whoever can write the
  config or the source can run code on the host. Treat the config as sensitive.
- Dependencies must already be installed in the backend environment.
- The plug-in path is **fail-closed** and supports **one** plug-in. A failed load aborts
  initialization — fix the config or source and restart.
- Plug-ins activate only at process/backend startup. Restart after changing the config
  or the source; there is no hot reload.

See [Plug-In Troubleshooting](./troubleshooting/plugins.md) for common failures.
