param(
    [string]$OpenSearchRoot = "D:\globex-runtime\opensearch-2.15.0"
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath($OpenSearchRoot)
$launcher = [IO.Path]::Combine($root, "bin", "opensearch.bat")
if (-not (Test-Path -LiteralPath $launcher)) {
    throw "OpenSearch runtime not found: $root"
}

try {
    $health = Invoke-RestMethod `
        -Uri "http://127.0.0.1:9200/_cluster/health" `
        -TimeoutSec 3
    Write-Host "OpenSearch already running: $($health.status)"
    exit 0
}
catch {
    # Start the local node below.
}

$env:OPENSEARCH_JAVA_HOME = [IO.Path]::Combine($root, "jdk")
$nativeLib = [IO.Path]::Combine(
    $root,
    "plugins",
    "opensearch-knn",
    "lib"
)
$env:PATH = $nativeLib + ";" + $env:PATH
$process = Start-Process `
    -FilePath $launcher `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput ([IO.Path]::Combine($root, "logs", "globex-stdout.log")) `
    -RedirectStandardError ([IO.Path]::Combine($root, "logs", "globex-stderr.log")) `
    -PassThru

for ($attempt = 1; $attempt -le 12; $attempt++) {
    try {
        $health = Invoke-RestMethod `
            -Uri "http://127.0.0.1:9200/_cluster/health" `
            -TimeoutSec 3
        Write-Host (
            "OpenSearch started: pid={0}, status={1}" -f
            $process.Id,
            $health.status
        )
        exit 0
    }
    catch {
        Start-Sleep -Seconds 5
    }
}

throw "OpenSearch did not become ready within 60 seconds."
