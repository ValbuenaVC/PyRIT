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
  - name: rapid_response
    source: /repos/pyrit-internal
    initializer: pyrit_internal.setup.initializers.RapidResponseInitializer
```

- `name` — an operator label used in logs and errors.
- `source` — a directory placed on `sys.path` so `import <your_package>` resolves. A
  relative path resolves against the config file.
- `initializer` — a dotted `module.Class` path to a concrete `PyRITInitializer`.

`ConfigurationLoader` prepends a privileged initializer that anchors `source`, imports
the initializer, and runs it before any user-configured initializer. Do **not** list it
under `initializers:`.

## The initializer owns registration

PyRIT discovers nothing on its own. The plug-in's initializer registers everything it
wants discoverable, at whatever level of abstraction fits:

- **Attack techniques** —
  `AttackTechniqueRegistry.get_registry_singleton().register_from_factories([...])`;
  selectable via `--techniques`.
- **Scenarios** —
  `ScenarioRegistry.get_registry_singleton().register_class(MyScenario, name="airt_internal.violence")`;
  runnable via `pyrit_scan airt_internal.violence`.
- **Datasets** — register providers and load them into memory so private seeds stay in
  the operator's database and are never published.
- **Default targets** — `set_default_value(...)`.

This is why the plug-in fits gray-area content: sometimes only the *dataset* must stay
private, sometimes a *technique*, and sometimes an entire *scenario* — the initializer
decides.

### Example initializer

```python
from pyrit.executor.attack import PromptSendingAttack
from pyrit.registry import AttackTechniqueRegistry, ScenarioRegistry
from pyrit.scenario.core import AttackTechniqueFactory
from pyrit.setup.pyrit_initializer import PyRITInitializer

from my_package.scenarios import Violence


class MyInitializer(PyRITInitializer):
    """Register a private technique and a private scenario."""

    async def initialize_async(self) -> None:
        AttackTechniqueRegistry.get_registry_singleton().register_from_factories(
            [AttackTechniqueFactory(name="operation_foobar", attack_class=PromptSendingAttack)]
        )
        ScenarioRegistry.get_registry_singleton().register_class(Violence, name="airt_internal.violence")
```

## Usage

```powershell
# A private technique through a public scenario
pyrit_scan airt.rapid_response --target openai_chat --techniques operation_foobar

# A private scenario
pyrit_scan airt_internal.violence --target openai_chat
```

## Behavior and limits

- The plug-in initializer runs **first**, so lazy catalog/metadata consumers see a
  complete registry.
- Loading executes third-party Python with backend permissions; whoever can write the
  config or the source can run code on the host. Treat the config as sensitive.
- Dependencies must already be installed in the backend environment.
- V1 is **fail-closed** and supports **one** plug-in. A failed load aborts
  initialization — fix the config or source and restart.
- Plug-ins activate only at process/backend startup. Restart after changing the config
  or the source; there is no hot reload.

See [Plug-In Troubleshooting](./troubleshooting/plugins.md) for common failures.
