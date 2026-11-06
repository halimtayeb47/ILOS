# Contributing

Start with a small, reproducible scenario. Explain the expected route, the actual
route and the command used to produce it.

```bash
python -m pip install -e './modern[test,api,learning]'
python -m pytest modern/tests -q
```

Changes to movement rules should cover vertex collisions, head-on swaps, stopped
vehicles, parked goals and timeouts. Keep the modern implementation independent
of MindSpore. Document any maintenance changes to the recovered source in
`huawei/README.md` and retain its input hashes in `huawei/source-manifest.json`.

Use measured results when comparing policies. A successful run on the training
map is not evidence of generalization. Do not commit local environments, private
forms, credentials or intermediate training runs.
