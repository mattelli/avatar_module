# Plan 5, task 2 — reproducible weight conversion pipeline (Windows).
#
# Steps:
#   1. Export PyTorch checkpoints -> ONNX (convert_weights.py)
#   2. Verify ONNX round-trips against PyTorch (verify_onnx.py)
#   3. Import ONNX -> Burn .bpk records via onnx2burn
#   4. Write SHA-256 checksums to .res/models/CHECKSUMS.txt
#
# Requires (in scripts/.venv-avatar):
#   torch, onnx, onnxscript, onnxruntime, numpy
# Plus the `onnx2burn` CLI (cargo install burn-onnx --version 0.22.0 --locked).

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Res = Join-Path $Root ".res"
$Py = Join-Path $Root "scripts\.venv-avatar\Scripts\python.exe"

# torch's ONNX exporter prints emoji (✅) which crashes the cp1252 console.
$env:PYTHONIOENCODING = "utf-8"

Write-Host "== Step 1: PyTorch -> ONNX =="
& $Py "$PSScriptRoot\convert_weights.py"
if ($LASTEXITCODE -ne 0) { throw "convert_weights.py failed" }

Write-Host "`n== Step 2: verify ONNX round-trip =="
& $Py "$PSScriptRoot\verify_onnx.py"
if ($LASTEXITCODE -ne 0) { throw "verify_onnx.py failed" }

Write-Host "`n== Step 3: ONNX -> Burn .bpk =="
$WavOut = Join-Path $Res "models\wav2lip"
$ExpOut = Join-Path $Res "models\expnet"
$PoseOut = Join-Path $Res "models\pose"
New-Item -ItemType Directory -Force -Path $WavOut, $ExpOut, $PoseOut | Out-Null
onnx2burn (Join-Path $Res "wav2lip_gan.onnx") $WavOut | Out-Null
if ($LASTEXITCODE -ne 0) { throw "onnx2burn wav2lip failed" }
onnx2burn (Join-Path $Res "expnet.onnx") $ExpOut | Out-Null
if ($LASTEXITCODE -ne 0) { throw "onnx2burn expnet failed" }
onnx2burn (Join-Path $Res "pose.onnx") $PoseOut | Out-Null
if ($LASTEXITCODE -ne 0) { throw "onnx2burn pose failed" }

Write-Host "`n== Step 4: checksums =="
$Checksums = @(
    "wav2lip_gan.pth",
    "auido2exp_00300-model.pth",
    "auido2pose_00140-model.pth",
    "wav2lip_gan.onnx",
    "expnet.onnx",
    "pose.onnx",
    "models\wav2lip\wav2lip_gan.bpk",
    "models\expnet\expnet.bpk",
    "models\pose\pose.bpk"
)
$lines = foreach ($f in $Checksums) {
    $p = Join-Path $Res $f
    if (Test-Path $p) {
        $h = (Get-FileHash -Path $p -Algorithm SHA256).Hash.ToLower()
        "$h  $f"
    }
}
$lines | Set-Content (Join-Path $Res "models\CHECKSUMS.txt")
$lines
Write-Host "`nWrote .res\models\CHECKSUMS.txt"
