
# Get the directory where this script is located (examples/)
$ScriptDir = $PSScriptRoot
# Calculate the project root (smlm_score/)
$ProjectRoot = Split-Path $ScriptDir -Parent
# Calculate the parent directory (Thesis/) for PYTHONPATH
$ThesisDir = Split-Path $ProjectRoot -Parent
$EnvRoot = "C:\envs\py311"
$PythonExe = Join-Path $EnvRoot "python.exe"

Write-Host "Setting up environment..."
Write-Host "Script Directory: $ScriptDir"
Write-Host "Thesis Directory (added to PYTHONPATH): $ThesisDir"
Write-Host "Python Environment: $EnvRoot"

if (-not (Test-Path $PythonExe)) {
    Write-Error "Python not found at $PythonExe"
    exit 1
}

# Add Thesis directory to PYTHONPATH so python can find 'smlm_score' package
$env:PYTHONPATH = "$ThesisDir;$env:PYTHONPATH"
$env:PATH = "$EnvRoot;$EnvRoot\Scripts;$EnvRoot\Library\bin;$env:PATH"
$env:VIRTUAL_ENV = $EnvRoot
$env:CONDA_PREFIX = $EnvRoot
$env:PYTHONHOME = ""
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

# Set location to the script directory so relative paths (like 'ShareLoc_Data/') work correctly
Set-Location $ScriptDir

# Run the python script using the project environment
& $PythonExe "NPC_example_BD.py"
