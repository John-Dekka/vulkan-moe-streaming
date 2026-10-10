# Start llama-server with MoE expert streaming (Windows).
#
#   .\scripts\run-server.ps1 -Model D:\models\GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf
#
# Then chat at http://127.0.0.1:8080/ or point any OpenAI-compatible app at http://127.0.0.1:8080/v1
# The defaults are for a 16 GB GPU and 32 GB of RAM; see the README ("Tuning") for other machines.
param(
    [Parameter(Mandatory = $true)][string]$Model,  # first .gguf file of the model
    [string]$Server = "",                          # llama-server.exe (default: next to this repo)
    [int]$CacheGB = 12,                            # RAM for the expert cache
    [int]$GpuTierMB = 4000,                        # VRAM for the hottest experts (0 = off)
    [int]$Prefetch = 4,                            # experts of the next layer to load early (0 = off)
    [int]$IoThreads = 16,                          # parallel disk reads
    [int]$Threads = 0,                             # CPU threads (default: physical cores)
    [int]$Context = 16384,
    [int]$Port = 8080,
    [string]$Device = "",                          # e.g. Vulkan0 when an integrated GPU is also listed
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Extra
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
if (-not $Server) {
    foreach ($c in @("$root\bin\llama-server.exe", "$root\llama.cpp\build\bin\llama-server.exe")) {
        if (Test-Path $c) { $Server = $c; break }
    }
}
if (-not $Server -or -not (Test-Path $Server)) { throw "llama-server.exe not found - pass -Server or build it first" }
if ($Threads -le 0) { $Threads = (Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum }

$env:LLAMA_MOE_STREAM_GPU_MIB  = "$GpuTierMB"
$env:LLAMA_MOE_STREAM_PREFETCH = "$Prefetch"

$srvArgs = @(
    "-m", $Model, "--host", "127.0.0.1", "--port", "$Port",
    "-ngl", "999", "-t", "$Threads", "-c", "$Context", "-fa", "on", "--jinja",
    "-b", "2048", "-ub", "2048", "--parallel", "1",
    # every expert layer is streamed: the experts stay on disk, the hot ones in RAM / VRAM
    "--cpu-moe", "--load-mode", "none",
    "--moe-stream-cache", "$CacheGB", "--moe-stream-direct", "--moe-stream-io-threads", "$IoThreads"
)
if ($Device) { $srvArgs += @("--device", $Device) }
if ($Extra) { $srvArgs += $Extra }

Write-Host "starting $Server"
Write-Host "  expert cache $CacheGB GB RAM, GPU tier $GpuTierMB MB VRAM, $Threads threads"
& $Server @srvArgs
