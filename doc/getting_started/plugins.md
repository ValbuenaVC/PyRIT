# Private Scenarios and Attack Techniques

PyRIT plug-ins make private scenarios and attack techniques behave like built-in
components in the standard backend, GUI, and `pyrit_scan` catalog.

## Plug-In or Python Dependency?

Use ordinary Python composition when you own the application:

```python
import pyrit

# Build and run your custom workflow directly.
```

That is the simplest option for notebooks, services, and tools that already control
their Python entry point.

Use a plug-in when operators need private components inside a **stock PyRIT
installation**:

- keep the shipped `pyrit_scan` command and backend;
- discover private components through normal catalog APIs;
- avoid a private PyRIT fork or wrapper CLI;
- activate components from `.pyrit_conf`;
- deploy an artifact/config instead of application glue.

## V1 Scope

V1 plug-ins contribute:

- concrete `Scenario` subclasses;
- configured `AttackTechniqueFactory` instances.

They do not contribute custom initializers or lifecycle hooks. PyRIT's privileged
`PluginInitializer` owns loading, discovery, validation, registration, and rollback.

Only one plug-in is supported at a time.

## Source Plug-In

Source is useful for live operations where scenarios or techniques already exist on
the backend filesystem.

Supported shapes:

- one importable `.py` file;
- one Python package directory containing `__init__.py`.

### Private attack technique

```python
# /opt/pyrit/operation_foobar.py
from pyrit.executor.attack import PromptSendingAttack
from pyrit.scenario import AttackTechniqueFactory


OPERATION_FOOBAR = AttackTechniqueFactory(
    name="operation_foobar",
    attack_class=PromptSendingAttack,
    strategy_tags=[
        "single_turn",
        "scenario:airt.rapid_response",
    ],
)
```

The `scenario:<registry-name>` tag declares where a directly discovered factory is
available. Construction tooling may replace this bridge with an explicit contribution
manifest in the future.

Configure it:

```yaml
plugins:
  - name: operation_foobar
    format: source
    source: /opt/pyrit/operation_foobar.py
```

Then restart the backend and run:

```powershell
pyrit_scan airt.rapid_response `
  --target openai_chat `
  --strategies operation_foobar
```

### Private scenario

A source scenario follows the normal `Scenario` contract:

- concrete subclass;
- keyword-only constructor;
- no-argument instantiable for catalog metadata;
- runtime dependencies resolved lazily;
- `_build_atomic_attacks_async` implemented.

The default registry name comes from the source-relative module path. For example:

```text
/opt/pyrit/private_scenarios/image/abuse.py -> image.abuse
```

A package directory must contain `__init__.py`; a directory of unrelated loose files
is rejected.

## Wheel Plug-In

Wheels are appropriate for durable distribution of multi-file private scenarios and
resources. The wheel must already be built and compatible with the running PyRIT.

```yaml
plugins:
  - name: partner_scenarios
    format: wheel
    wheel: /opt/pyrit/partner_scenarios-1.2.0-py3-none-any.whl
    package: partner_scenarios
```

PyRIT safely extracts the wheel to `.plugin/`; it does not run `pip`, install into
`.venv`, or resolve dependencies.

If wheel metadata declares another PyRIT version, loading emits an advisory warning.
PyRIT does not rewrite incompatible APIs. Rebuild the wheel against the running
version when validation fails.

## Initialization Behavior

`ConfigurationLoader` converts the config entry into a privileged
`PluginInitializer`. Operators do not add this initializer to `initializers:`.

Runtime order:

1. load environment;
2. create CentralMemory;
3. activate the configured plug-in;
4. execute user-configured initializers;
5. serve catalog and scenario requests.

This ordering ensures scenario metadata and technique strategies cannot be built from
a partial catalog.

For direct Python use:

```python
from pyrit.setup import initialize_from_config_async

await initialize_from_config_async("/path/to/.pyrit_conf")
```

Low-level `initialize_pyrit_async()` does not read `.pyrit_conf` automatically.

## Trust Boundary

Source and wheel plug-ins execute Python with backend permissions. Treat write access
to the artifact and `.pyrit_conf` as code-execution authority.

Dependencies must already be installed in the backend environment. Plug-ins share one
interpreter and do not receive dependency isolation.

## Updating a Plug-In

V1 does not hot reload.

```powershell
pyrit_scan --stop-server
pyrit_scan --start-server --config-file /path/to/.pyrit_conf
```

For direct Python use, initialize in a fresh process after changing the artifact.

See [Plug-In Troubleshooting](./troubleshooting/plugins.md) for common failures.
