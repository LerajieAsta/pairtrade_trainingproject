param([string]$Path)
# Word pass: center tables, update TOC / list of tables, save, export PDF.
$wd = New-Object -ComObject Word.Application
$wd.Visible = $false
$wd.DisplayAlerts = 0
try {
  $doc = $wd.Documents.Open($Path, $false, $false)
  foreach ($t in $doc.Tables) {
    $t.Rows.Alignment = 1              # wdAlignRowCenter
  }
  $doc.Repaginate()
  foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
  $doc.Fields.Update() | Out-Null
  foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
  $pages = $doc.ComputeStatistics(2)
  $ntab = $doc.Tables.Count
  $doc.Save()
  $pdf = [System.IO.Path]::ChangeExtension($Path, ".pdf")
  $doc.ExportAsFixedFormat($pdf, 17)
  $doc.Close($false)
  "Word: $pages pages, $ntab tables"
} catch {
  Write-Error $_
  exit 1
} finally {
  $wd.Quit()
  [System.Runtime.Interopservices.Marshal]::ReleaseComObject($wd) | Out-Null
}
