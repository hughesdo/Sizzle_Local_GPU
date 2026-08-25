# fix-certs.ps1 - make Python trust HTTPS on this box.
#
# Why this exists: Avast antivirus does TLS/SSL scanning (a man-in-the-middle).
# Its "Avast Web/Mail Shield Root" sits in the Windows cert store, so browsers
# and the C# RightClicks app trust it - but Python's bundled certifi does not,
# so every HTTPS call (Anthropic auto-prompt, Hugging Face model downloads) fails with
# CERTIFICATE_VERIFY_FAILED. This appends the Windows trusted roots (incl. the
# Avast root) into certifi's cacert.pem, which requests/httpx/urllib all use.
#
# Idempotent: keeps a one-time .orig backup and rebuilds from it each run, so
# re-running never double-appends. Re-run this after any `pip install certifi`
# or Python upgrade, which would overwrite the patched bundle.

$ErrorActionPreference = "Stop"
$certifi = & python -c "import certifi; print(certifi.where())"
if (-not $certifi) { throw "could not locate certifi (is it installed?)" }
Write-Host "certifi bundle: $certifi"

if (-not (Test-Path "$certifi.orig")) { Copy-Item $certifi "$certifi.orig" }
Copy-Item "$certifi.orig" $certifi -Force   # start from pristine each run

$sb = New-Object System.Text.StringBuilder
$roots = Get-ChildItem Cert:\LocalMachine\Root, Cert:\CurrentUser\Root -ErrorAction SilentlyContinue |
         Sort-Object Thumbprint -Unique
foreach ($c in $roots) {
  $b64 = [System.Convert]::ToBase64String($c.RawData, 'InsertLineBreaks')
  [void]$sb.AppendLine(""); [void]$sb.AppendLine("# " + $c.Subject)
  [void]$sb.AppendLine("-----BEGIN CERTIFICATE-----")
  [void]$sb.AppendLine($b64)
  [void]$sb.AppendLine("-----END CERTIFICATE-----")
}
Add-Content -Path $certifi -Value $sb.ToString() -Encoding ascii
Write-Host "appended $($roots.Count) Windows roots into certifi"
Write-Host ("Avast root present: " + ((Get-Content -Raw $certifi) -match 'Avast'))
