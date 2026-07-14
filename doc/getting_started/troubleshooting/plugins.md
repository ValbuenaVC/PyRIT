# Plug-In Troubleshooting

## The plug-in did not activate

Restart the backend after changing `.pyrit_conf` or the source:

```powershell
pyrit_scan --stop-server
pyrit_scan --start-server --config-file /path/to/.pyrit_conf
```

`pyrit_scan` is a thin client. An already-running backend keeps the plug-in loaded from
its startup config.

## Configuration is rejected

V1 accepts exactly one entry with three fields:

```yaml
plugins:
  - name: rapid_response
    source: /repos/pyrit-internal
    initializer: pyrit_internal.setup.initializers.RapidResponseInitializer
```

Check that:

- `name` is a valid lowercase snake_case registry name;
- `source` is present (relative paths resolve against the config file);
- `initializer` is a dotted `module.Class` path;
- only one plug-in is configured.

## Source path does not exist

The loader fails closed with a "source path does not exist" error. Point `source` at the
directory that contains your package (the parent of the top-level package) so
`import <your_package>` resolves once that directory is on `sys.path`.

## Initializer cannot be imported

- Confirm `initializer` names a real `module.Class` importable from `source`.
- Install the plug-in's dependencies into the backend Python environment.
- A packaged initializer must import from its own package (for example
  `from my_package... import ...`); pointing `source` at the package root makes that
  work.

## Target is not a PyRITInitializer

`initializer` must resolve to a concrete subclass of `PyRITInitializer`. A plug-in
contributes an initializer, not loose scenario or technique objects.

## Nothing shows up in the catalog

PyRIT does not discover components — the initializer must register them. For scanner
discovery, the initializer must call
`ScenarioRegistry.get_registry_singleton().register_class(...)` for scenarios and
`AttackTechniqueRegistry.get_registry_singleton().register_from_factories(...)` for
techniques. Datasets must be registered as providers and loaded into memory.

## Partial state after a failed load

V1 is fail-closed. A failed plug-in load aborts initialization; fix the reported stage
and restart the process. Continuing in the same process after a failed initialization is
not supported.
