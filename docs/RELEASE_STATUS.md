# Release status

The prepared source has not been pushed to GitHub. The connected integration
returned HTTP 403, `Resource not accessible by integration`, when creating a
repository blob. The repository is already public. The connected user has write
permission, but this does not grant the integration access to the owner's repo.

The repository owner needs to enable the GitHub integration for
`halimtayeb47/ILOS`. Once access is available, this package can be committed on top
of the inspected `main` commit, `7cb2953b3200e14be9275063483198978bdf4757`.

## Optional local publishing

If your local Git installation already has access to the repository, run this
from the extracted project root:

```bash
python scripts/publish.py
```

It clones the existing repository, checks that `main` has not changed, copies only
the manifest's release files, removes the duplicate old README, commits with your
configured Git identity and pushes `release/ilos-0.2`. It prints a comparison link
for review and merge. It does not change repository visibility, force-push or
rewrite existing contributors. GitHub workflows can run after the push.

## Verification completed

- 32 local tests passed, including real PPO training and checkpoint reload.
- Trained PPO: 3/3 arrivals in 14 steps, zero conflict proposals or blocked moves.
- CBS: all six vehicles routed with zero conflicts, makespan 19 steps.
- Browser: playback, pause, restart and timeline work; mobile layout has no
  horizontal overflow; no JavaScript errors observed.
- Source and checkpoint metadata scanned for credential-like strings and unwanted
  assistant branding. No matching credentials or branding found.
- Existing GitHub history contained two human-authored commits and duplicated
  README content; no assistant author was present.

## Verification still pending

- Windows execution: CI is configured, but no workflow could be run without
  repository write access.
- MindSpore execution: native imports crashed with a bus error in this container
  under both 2.7.2 and 2.10.0. The port is clearly marked; its separate CI job is
  pending. Original MindSpore source and weights were not in the upload.
