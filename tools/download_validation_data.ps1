param(
    [string]$DataDir = "..\..\data\zenodo_3825933"
)

$ErrorActionPreference = "Stop"

# Resolve output directory relative to the repository root.
$RepoRoot = Split-Path -Parent $PSScriptRoot
$OutputDir = [System.IO.Path]::GetFullPath(
    (Join-Path $RepoRoot $DataDir)
)

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "Dataset directory:"
Write-Host "  $OutputDir"
Write-Host ""

$Files = @(
    @{
        Name = "val_dataset_1_norm.zip"
        Url  = "https://zenodo.org/records/3825933/files/val_dataset_1_norm.zip?download=1"
        MD5  = "1dda32f59996640d701b591f69353115"
    },
    @{
        Name = "val_dataset_1_tu.zip"
        Url  = "https://zenodo.org/records/3825933/files/val_dataset_1_tu.zip?download=1"
        MD5  = "35494acbfea3adbc689cc15f73a9116d"
    },
    @{
        Name = "val_dataset_2_norm.zip"
        Url  = "https://zenodo.org/records/3825933/files/val_dataset_2_norm.zip?download=1"
        MD5  = "029069ab405a76d42fcdefac32982989"
    },
    @{
        Name = "val_dataset_2_tu.zip"
        Url  = "https://zenodo.org/records/3825933/files/val_dataset_2_tu.zip?download=1"
        MD5  = "d72b78d364fbfed95d8f0d71d8f2d01e"
    }
)

foreach ($File in $Files) {

    $Destination = Join-Path $OutputDir $File.Name

    Write-Host "============================================================"
    Write-Host "Processing $($File.Name)"
    Write-Host "============================================================"

    # If a complete, valid file already exists, do not download again.
    if (Test-Path $Destination) {

        Write-Host "Existing file found. Checking MD5..."

        $ExistingHash = (
            Get-FileHash $Destination -Algorithm MD5
        ).Hash.ToLower()

        if ($ExistingHash -eq $File.MD5) {
            Write-Host "MD5 OK - skipping download."
            Write-Host ""
            continue
        }

        Write-Host "Existing file is incomplete or has the wrong MD5."
        Write-Host "curl will attempt to resume it."
    }

    Write-Host "Downloading/resuming..."

    & curl.exe `
        -L `
        -C - `
        --retry 5 `
        --retry-delay 5 `
        -o $Destination `
        $File.Url

    if ($LASTEXITCODE -ne 0) {
        throw "Download failed for $($File.Name)"
    }

    Write-Host "Checking MD5..."

    $ActualHash = (
        Get-FileHash $Destination -Algorithm MD5
    ).Hash.ToLower()

    if ($ActualHash -ne $File.MD5) {
        throw @"
MD5 verification FAILED for $($File.Name)

Expected: $($File.MD5)
Actual:   $ActualHash
"@
    }

    Write-Host "MD5 OK: $ActualHash"
    Write-Host ""
}

Write-Host "============================================================"
Write-Host "All four validation archives downloaded and verified."
Write-Host "Dataset directory:"
Write-Host "  $OutputDir"
Write-Host "============================================================"