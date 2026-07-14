# Plug-In Troubleshooting

## The plug-in is missing from `pyrit_scan`

Restart the backend after changing `.pyrit_conf` or the artifact:

```powershell
pyrit_scan --stop-server
pyrit_scan --start-server --config-file /path/to/.pyrit_conf
```

`pyrit_scan` is a thin client. An already-running backend keeps the plug-ins loaded
from its startup config.

## Configuration is rejected

V1 accepts one explicit entry:

```yaml
plugins:
  - name: operation_foobar
    format: source
    source: /absolute/path/operation_foobar.py
```

or:

```yaml
plugins:
  - name: partner_scenarios
    format: wheel
    wheel: /absolute/path/partner.whl
    package: partner_scenarios
```

Check that:

- `name` is lowercase snake case;
- `format` is `source` or `wheel`;
- exactly one matching artifact field is present;
- only one plug-in is configured.

Relative paths resolve against the configuration file.

## Source path is rejected

A source path must be:

- one `.py` file with an importable filename; or
- one package directory containing `__init__.py`.

Loose directories containing unrelated Python files are not supported.

## Import fails

Plug-in dependencies are not installed automatically. Install them into the backend
environment and restart.

Also verify:

- package imports work from the configured source root;
- the configured package name matches the wheel/source package;
- another installed package is not shadowing the plug-in.

## Scenario is rejected

The scenario must:

- inherit from `Scenario`;
- be concrete;
- use a keyword-only constructor;
- construct with no arguments for catalog metadata;
- implement `_build_atomic_attacks_async`.

Plug-in scenarios are inspected before user-configured target/scorer/technique
initializers run. Keep import and no-argument construction side-effect-light and defer
runtime dependency resolution.

## Technique is rejected

Expose factories through either:

- a module-owned `get_technique_factories()` returning
  `AttackTechniqueFactory` instances; or
- module-global `AttackTechniqueFactory` instances.

Directly discovered factories must identify an applicable scenario, for example:

```python
technique_tags=["single_turn", "scenario:airt.rapid_response"]
```

Do not register the factory during module import. The framework owns registration and
rollback.

## Name collision

V1 is extend-only. A private scenario or technique cannot replace a built-in or
existing registry name. Rename the contribution and restart.

## Version drift warning

The warning is advisory. PyRIT validates the live scenario/factory contract but does
not patch incompatible code. Rebuild or update the artifact against the installed
PyRIT version.

## Partial state after failure

Plug-in activation is fail-closed and transactional for supported scenario/technique
registries. If initialization fails, fix the reported stage and restart the process.

Continuing production work in the same process after failed initialization is not
supported.
