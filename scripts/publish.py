"""Push this prepared source to a review branch using your local Git credentials."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

REPOSITORY = 'https://github.com/halimtayeb47/ILOS.git'
BASE_COMMIT = '7cb2953b3200e14be9275063483198978bdf4757'
ROOT = Path(__file__).resolve().parents[1]


def git(*args, cwd=None):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch', default='release/ilos-0.2')
    args = parser.parse_args()
    if not shutil.which('git'):
        raise SystemExit('Install Git and authenticate with GitHub first.')
    git('check-ref-format', '--branch', args.branch)
    if args.branch in ('main', 'master'):
        raise SystemExit('Choose a review branch rather than the default branch.')
    manifest = (ROOT / 'RELEASE_FILES.txt').read_text(encoding='utf-8').splitlines()
    with tempfile.TemporaryDirectory(prefix='ilos-release-') as temp:
        checkout = Path(temp) / 'repository'
        subprocess.run(['git', 'clone', '--branch', 'main', REPOSITORY, str(checkout)], check=True)
        if git('rev-parse', 'HEAD', cwd=checkout) != BASE_COMMIT:
            raise SystemExit('main changed after this package was prepared. Review the new changes before applying the package.')
        name = git('config', 'user.name', cwd=checkout)
        email = git('config', 'user.email', cwd=checkout)
        if not name or not email:
            raise SystemExit('Configure your Git user.name and user.email before publishing.')
        git('switch', '-c', args.branch, cwd=checkout)
        duplicate = checkout / 'ILOS_Complete/README.md'
        if duplicate.exists():
            duplicate.unlink()
        for relative in manifest:
            source = ROOT / relative
            if source.is_symlink() or not source.resolve().is_relative_to(ROOT):
                raise SystemExit(f'Unsafe release path: {relative}')
            target = checkout / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        git('add', '--all', cwd=checkout)
        git('diff', '--cached', '--check', cwd=checkout)
        git('commit', '-m', 'Organize ILOS and add trained PPO and Huawei backends', cwd=checkout)
        subprocess.run(['git', 'push', '--set-upstream', 'origin', args.branch], cwd=checkout, check=True)
    print('Uploaded review branch. Review and merge at:')
    print(f'https://github.com/halimtayeb47/ILOS/compare/main...{args.branch}')

if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f'Git command failed (exit {exc.returncode}). Check repository access and local Git authentication.') from exc
