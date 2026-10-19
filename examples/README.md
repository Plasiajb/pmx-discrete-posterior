# Examples by package version

For **0.2.0**, start with `posterior_reference.py` and `reference_models.py`.
They calculate the OXC/TAC reference posteriors using the seven-group API.

The other scripts and the `manuscript_current/` and `manuscript_demos/`
directories belong to the earlier manuscript implementation. They have not
been migrated to 0.2.0. Reproduce those results in a separate checkout and
environment:

```bash
git clone --branch manuscript-20261018 https://github.com/Plasiajb/pmx-discrete-posterior.git manuscript-replay
cd manuscript-replay
```

Then follow `examples/manuscript_current/README.md` there, using Python 3.12
and its pinned requirements. The tag resolves to
`408f28f4812040db5ad553c6295041e3bd2ecd66`.
Access to this private repository requires authorization.

The manuscript's complete sensitivity examples are retained at that tag;
they are not evidence that the full matrix has been rerun with 0.2.0.
