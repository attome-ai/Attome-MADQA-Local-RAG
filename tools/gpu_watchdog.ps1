# Watches the RTX 5090. Logs memory every minute, and stops MADQA GPU jobs if Windows starts spilling
# GPU memory into system RAM ("shared GPU memory"), which makes everything crawl.
#   powershell -File tools/gpu_watchdog.ps1 >> data/results/gpu_watchdog.log
param([double]$LimitGB = 0.4, [int]$Seconds = 10, [int]$LogEvery = 6)
$n = 0
while ($true) {
    $shared = ((Get-Counter '\GPU Adapter Memory(*)\Shared Usage').CounterSamples | Measure-Object CookedValue -Maximum).Maximum / 1GB
    $dedicated = ((Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage').CounterSamples | Measure-Object CookedValue -Maximum).Maximum / 1GB
    if ($shared -gt $LimitGB) {
        "{0} OVERFLOW: dedicated {1:N2} GiB, shared {2:N2} GiB; stopping GPU jobs" -f (Get-Date -Format T), $dedicated, $shared
        docker ps --format '{{.Names}}' | Where-Object { $_ -match '^madqa-(agent|job)' } | ForEach-Object { docker kill $_ | Out-Null; "killed $_" }
        # Jobs run from the host too (python -m lab.*); stop those, but never the vLLM server.
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match '-m lab\.' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "killed python $($_.ProcessId)" }
    } elseif ($n % $LogEvery -eq 0) {
        "{0} ok: dedicated {1:N2} of 31.84 GiB (spill starts ~30.9), shared {2:N2} GiB" -f (Get-Date -Format T), $dedicated, $shared
    }
    $n++
    Start-Sleep -Seconds $Seconds
}
