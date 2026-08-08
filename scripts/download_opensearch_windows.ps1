param(
    [string]$Version = "2.15.0",
    [string]$DestinationDirectory = "D:\globex-runtime",
    [int64]$ExpectedBytes = 839252526,
    [int64]$ChunkBytes = 25000000,
    [int]$Parallelism = 8
)

$ErrorActionPreference = "Stop"
$destination = [IO.Path]::GetFullPath($DestinationDirectory)
[IO.Directory]::CreateDirectory($destination) | Out-Null
$archiveName = "opensearch-$Version-windows-x64.zip"
$archivePath = [IO.Path]::Combine($destination, $archiveName)
$partsDirectory = [IO.Path]::Combine(
    $destination,
    "opensearch-$Version.parts"
)
[IO.Directory]::CreateDirectory($partsDirectory) | Out-Null
$uri = (
    "https://artifacts.opensearch.org/releases/bundle/" +
    "opensearch/$Version/$archiveName"
)

$parts = @()
$partIndex = 0
for ($start = [int64]0; $start -lt $ExpectedBytes; $start += $ChunkBytes) {
    $end = [Math]::Min($ExpectedBytes - 1, $start + $ChunkBytes - 1)
    $path = [IO.Path]::Combine(
        $partsDirectory,
        ("part-{0:D4}.bin" -f $partIndex)
    )
    $parts += [pscustomobject]@{
        Index = $partIndex
        Start = $start
        End = $end
        Path = $path
        Length = $end - $start + 1
    }
    $partIndex++
}

$pending = [Collections.Generic.Queue[object]]::new()
foreach ($part in $parts) {
    if (
        (Test-Path -LiteralPath $part.Path) -and
        (Get-Item -LiteralPath $part.Path).Length -eq $part.Length
    ) {
        continue
    }
    $pending.Enqueue($part)
}

$jobs = @()
while ($pending.Count -gt 0 -or $jobs.Count -gt 0) {
    while ($pending.Count -gt 0 -and $jobs.Count -lt $Parallelism) {
        $part = $pending.Dequeue()
        $job = Start-Job -ArgumentList @(
            $uri,
            $part.Start,
            $part.End,
            $part.Path
        ) -ScriptBlock {
            param($downloadUri, $rangeStart, $rangeEnd, $outputPath)
            & curl.exe `
                --silent `
                --show-error `
                --location `
                --retry 8 `
                --retry-all-errors `
                --connect-timeout 20 `
                --max-time 300 `
                --range "$rangeStart-$rangeEnd" `
                --output $outputPath `
                $downloadUri
            if ($LASTEXITCODE -ne 0) {
                throw "curl failed for $rangeStart-$rangeEnd"
            }
        }
        $job | Add-Member -NotePropertyName Part -NotePropertyValue $part
        $jobs += $job
    }

    $completed = Wait-Job -Job $jobs -Any -Timeout 10
    if ($null -eq $completed) {
        continue
    }
    Receive-Job -Job $completed -ErrorAction SilentlyContinue | Out-Null
    if ($completed.State -ne "Completed") {
        throw "Chunk download failed: $($completed.Part.Index)"
    }
    $part = $completed.Part
    $actualLength = (Get-Item -LiteralPath $part.Path).Length
    if ($actualLength -ne $part.Length) {
        throw (
            "Invalid chunk length: {0}, expected {1}, actual {2}" -f
            $part.Index, $part.Length, $actualLength
        )
    }
    Remove-Job -Job $completed
    $jobs = @($jobs | Where-Object { $_.Id -ne $completed.Id })
    Write-Host (
        "Downloaded {0}/{1} chunks" -f
        ($parts.Count - $pending.Count - $jobs.Count),
        $parts.Count
    )
}

$output = [IO.File]::Open(
    $archivePath,
    [IO.FileMode]::Create,
    [IO.FileAccess]::Write,
    [IO.FileShare]::None
)
try {
    foreach ($part in $parts | Sort-Object Index) {
        $input = [IO.File]::OpenRead($part.Path)
        try {
            $input.CopyTo($output)
        }
        finally {
            $input.Dispose()
        }
    }
}
finally {
    $output.Dispose()
}

$archiveLength = (Get-Item -LiteralPath $archivePath).Length
if ($archiveLength -ne $ExpectedBytes) {
    throw "Invalid archive length: $archiveLength != $ExpectedBytes"
}
foreach ($part in $parts) {
    Remove-Item -LiteralPath $part.Path -Force
}
Remove-Item -LiteralPath $partsDirectory -Force
Write-Host "OpenSearch download completed: $archivePath"
