# Merge notes

## Sources

- Implementation baseline: `/home/jqliu/sad` current working tree.
- Additive source: `/home/rzzhang/sad` current working tree.
- Merge destination: `/home/rzzhang/sad_merged`.

## Merge policy

1. Existing SAD circuit implementations from `jqliu` remain authoritative.
2. Files unique to `rzzhang` are added without replacing same-path files.
3. The new MERA, equivariant-QNN, data-reuploading, and QAOA-NS circuits are
   registered through the shared API, dispatch, runtime, and Python layers.
4. Shared runtime changes retain `jqliu` phase-plan and kernel-selection support.
5. Generated environments and artifacts are not copied: `.venv`, `build`,
   `.pytest_cache`, `__pycache__`, and `*.egg-info`.

## Verification

```bash
make -C sad
/home/jqliu/sad/.venv/bin/pytest -q sad/tests pennylane-lightning/tests
```

The pre-existing circuit headers can be checked against the source with
`sha256sum` or `cmp` under `sad/src/circuits`.
