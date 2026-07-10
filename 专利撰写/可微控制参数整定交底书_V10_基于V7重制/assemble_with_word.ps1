param(
    [Parameter(Mandatory = $true)][string]$ShellPath,
    [Parameter(Mandatory = $true)][string]$BodyPath,
    [Parameter(Mandatory = $true)][string]$OutputPath
)

$ErrorActionPreference = "Stop"
$word = $null
$shell = $null
$body = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3

    $shell = $word.Documents.Open($ShellPath, $false, $false)
    $body = $word.Documents.Open($BodyPath, $false, $true)

    $target = $shell.Content.Duplicate
    $found = $target.Find.Execute("[[BODY_INSERT]]")
    if (-not $found) {
        throw "Body insertion marker was not found."
    }
    $target = $target.Paragraphs.Item(1).Range

    $source = $body.Content.Duplicate
    if ($source.End -gt $source.Start) {
        $source.End = $source.End - 1
    }
    $target.FormattedText = $source.FormattedText

    $shell.Fields.Update() | Out-Null
    foreach ($section in $shell.Sections) {
        $section.Footers.Item(1).Range.Fields.Update() | Out-Null
    }
    $shell.SaveAs2($OutputPath, 16)
}
finally {
    if ($body -ne $null) {
        $body.Close($false)
    }
    if ($shell -ne $null) {
        $shell.Close($false)
    }
    if ($word -ne $null) {
        $word.Quit()
    }
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}
