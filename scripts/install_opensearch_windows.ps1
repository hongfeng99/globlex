param(
    [string]$Version = "2.15.0",
    [string]$DestinationDirectory = "D:\globex-runtime"
)

$ErrorActionPreference = "Stop"
$destination = [IO.Path]::GetFullPath($DestinationDirectory)
$root = [IO.Path]::Combine($destination, "opensearch-$Version")
$launcher = [IO.Path]::Combine($root, "bin", "opensearch.bat")
$archive = [IO.Path]::Combine(
    $destination,
    "opensearch-$Version-windows-x64.zip"
)

if (-not (Test-Path -LiteralPath $launcher)) {
    if (-not (Test-Path -LiteralPath $archive)) {
        & "$PSScriptRoot\download_opensearch_windows.ps1" `
            -Version $Version `
            -DestinationDirectory $destination
    }
    $checksumUri = (
        "https://artifacts.opensearch.org/releases/bundle/" +
        "opensearch/$Version/opensearch-$Version-windows-x64.zip.sha512"
    )
    $checksumPath = $archive + ".sha512"
    & curl.exe `
        --silent `
        --show-error `
        --location `
        --output $checksumPath `
        $checksumUri
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to download OpenSearch checksum."
    }
    $expected = (
        Get-Content -LiteralPath $checksumPath -Raw
    ).Substring(0, 128).ToUpperInvariant()
    $actual = (
        Get-FileHash -LiteralPath $archive -Algorithm SHA512
    ).Hash
    if ($actual -ne $expected) {
        throw "OpenSearch SHA512 mismatch."
    }
    Expand-Archive `
        -LiteralPath $archive `
        -DestinationPath $destination `
        -Force
}

$configPath = [IO.Path]::Combine($root, "config", "opensearch.yml")
$config = Get-Content -LiteralPath $configPath -Raw
if (-not $config.Contains("cluster.name: globex-local")) {
    $config += @"

# Globex local-only development node.
cluster.name: globex-local
node.name: globex-local-node
network.host: 127.0.0.1
http.port: 9200
discovery.type: single-node
plugins.security.disabled: true
action.destructive_requires_name: true
"@
    Set-Content -LiteralPath $configPath -Value $config -Encoding UTF8
}

$jvmPath = [IO.Path]::Combine($root, "config", "jvm.options")
$jvm = Get-Content -LiteralPath $jvmPath -Raw
$jvm = $jvm.Replace("-Xms1g", "-Xms512m")
$jvm = $jvm.Replace("-Xmx1g", "-Xmx512m")
Set-Content -LiteralPath $jvmPath -Value $jvm -Encoding UTF8
Write-Host "OpenSearch local runtime installed: $root"
