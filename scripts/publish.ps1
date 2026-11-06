$ErrorActionPreference = 'Stop'
python (Join-Path $PSScriptRoot 'publish.py') @args
exit $LASTEXITCODE
