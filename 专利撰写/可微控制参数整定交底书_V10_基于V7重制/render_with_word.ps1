param(
    [Parameter(Mandatory = $true)][string]$InputPath,
    [Parameter(Mandatory = $true)][string]$OutputPath
)

$ErrorActionPreference = "Stop"
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $document = $word.Documents.Open($InputPath, $false, $true, $false, "", "", $false, "", "", 0, 0, $false, $true, 0, $true, "")
    $document.SaveAs2($OutputPath, 17)
}
finally {
    if ($document -ne $null) {
        $document.Close($false)
    }
    if ($word -ne $null) {
        $word.Quit()
    }
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}
