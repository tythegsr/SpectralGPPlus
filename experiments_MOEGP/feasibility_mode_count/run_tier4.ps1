# Tier-4 benchmark driver: launches every job as a background process.
#   powershell -File run_tier4.ps1 [-Only main|real|fresh|large|refs]
# GP jobs run CPU-only, 1 thread each. TabPFN references run in ONE sequential
# GPU process (concurrent TabPFN loads exhaust the 12 GB card) and write to
# refs_* dirs, which the aggregator merges by (problem, n_train, seed).
param([string]$Only = "")

$ErrorActionPreference = "Stop"
$py = "C:\Users\forty\anaconda3\envs\gpplus2026\python.exe"
$root = "results/foundation_compare_tier4"
$logs = "$root/logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$env:PYTHONIOENCODING = "utf-8"

$synth = "A01,unimodal,A02,A04,A05,A06"
$main = @(
    "exact_gp",
    "midline_gamma_auto", "midline_gamma_auto_c85", "midline_gamma_auto_calibaccept",
    "midline_gamma_auto_sorf",
    "logistic_gp_sorf_map", "logistic_gp_sorf_vi", "logistic_gp_sorf_gamma",
    "quantile_gp_sorf", "quantile_gp_hybrid", "quantile_gp_hybrid_l01", "quantile_gp_hybrid_auto",
    "hist_mogp_t1"
) -join ","
$fresh = "exact_gp,midline_gamma_auto,midline_gamma_auto_c85,midline_gamma_auto_calibaccept"
$large = "midline_gamma_auto_c85,midline_gamma_auto_sorf,logistic_gp_sorf_map,quantile_gp_sorf"

function Launch([string]$name, [string[]]$argv) {
    $env:CUDA_VISIBLE_DEVICES = ""
    $env:OMP_NUM_THREADS = "1"
    $env:MKL_NUM_THREADS = "1"
    $p = Start-Process -FilePath $py -ArgumentList (@("compare_foundation_baselines.py") + $argv) `
        -RedirectStandardOutput "$logs/$name.log" -RedirectStandardError "$logs/$name.err" `
        -NoNewWindow -PassThru
    Write-Output "$name pid=$($p.Id)"
}

if ($Only -in @("", "main")) {
    foreach ($s in 0, 1, 42) {
        Launch "main_seed$s" @("--problems", $synth, "--models", $main, "--train-sizes", "100",
            "--n-test", "200", "--seed", "$s", "--out-dir", "$root/seed$s")
    }
}
if ($Only -in @("", "real")) {
    foreach ($s in 0, 1, 42) {
        foreach ($p in "QSAR_FISH", "TABARENA_CONCRETE") {
            Launch "real_$($p.ToLower())_seed$s" @("--problems", $p, "--models", $main,
                "--seed", "$s", "--out-dir", "$root/real_$($p.ToLower())_seed$s")
        }
    }
}
if ($Only -in @("", "fresh")) {
    foreach ($s in 3, 4, 5) {
        Launch "fresh_seed$s" @("--problems", $synth, "--models", $fresh, "--train-sizes", "100",
            "--n-test", "200", "--seed", "$s", "--no-marginal-plots", "--out-dir", "$root/fresh_seed$s")
    }
}
if ($Only -in @("", "large")) {
    foreach ($p in "A01", "unimodal", "A02", "A05", "A06") {
        Launch "large1000_$($p.ToLower())" @("--problems", $p, "--models", "$large,exact_gp",
            "--train-sizes", "1000", "--n-test", "1000", "--seed", "0", "--out-dir", "$root/large_n1000")
        Launch "large5000_$($p.ToLower())" @("--problems", $p, "--models", $large,
            "--train-sizes", "5000", "--n-test", "1000", "--seed", "0", "--out-dir", "$root/large_n5000")
    }
}
if ($Only -in @("", "refs")) {
    # Sequential on the GPU, in its own background PowerShell.
    $cmds = @()
    foreach ($s in 0, 1, 42, 3, 4, 5) {
        $cmds += "& '$py' compare_foundation_baselines.py --problems $synth --models tabpfn35 --train-sizes 100 --n-test 200 --seed $s --no-marginal-plots --out-dir $root/refs_seed$s"
    }
    foreach ($s in 0, 1, 42) {
        foreach ($p in "QSAR_FISH", "TABARENA_CONCRETE") {
            $cmds += "& '$py' compare_foundation_baselines.py --problems $p --models tabpfn35 --seed $s --no-marginal-plots --out-dir $root/refs_$($p.ToLower())_seed$s"
        }
    }
    foreach ($p in "A01", "unimodal", "A02", "A05", "A06") {
        foreach ($n in 1000, 5000) {
            $cmds += "& '$py' compare_foundation_baselines.py --problems $p --models tabpfn35 --train-sizes $n --n-test 1000 --seed 0 --no-marginal-plots --out-dir $root/refs_large_n$n"
        }
    }
    $script = "`$env:PYTHONIOENCODING='utf-8'; `$env:CUDA_VISIBLE_DEVICES='0'; " + ($cmds -join "; ")
    $p = Start-Process -FilePath "powershell" -ArgumentList @("-NoProfile", "-Command", $script) `
        -RedirectStandardOutput "$logs/refs.log" -RedirectStandardError "$logs/refs.err" `
        -NoNewWindow -PassThru
    Write-Output "refs pid=$($p.Id)"
}
