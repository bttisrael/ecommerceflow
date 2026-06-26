param(
    [string]$ProjectId = $env:PROJECT_ID,
    [string]$Region = $(if ($env:REGION) { $env:REGION } else { "us-central1" }),
    [string]$EndpointId = $env:VERTEX_ENDPOINT_ID,
    [switch]$Apply,
    [switch]$Force
)

if (-not $ProjectId) {
    throw "ProjectId is required. Pass -ProjectId or set PROJECT_ID."
}

if (-not $EndpointId) {
    throw "EndpointId is required. Pass -EndpointId or set VERTEX_ENDPOINT_ID."
}

$endpoint = gcloud ai endpoints describe $EndpointId `
    --project=$ProjectId `
    --region=$Region `
    --format=json | ConvertFrom-Json

if (-not $endpoint.deployedModels -or $endpoint.deployedModels.Count -eq 0) {
    Write-Host "No deployed models found on endpoint $EndpointId in $Region."
    exit 0
}

Write-Host "Endpoint: $EndpointId"
Write-Host "Project : $ProjectId"
Write-Host "Region  : $Region"
Write-Host ""
Write-Host "Deployed models:"

foreach ($model in $endpoint.deployedModels) {
    Write-Host ("- id={0} displayName={1}" -f $model.id, $model.displayName)
}

if (-not $Apply) {
    Write-Host ""
    Write-Host "Dry run only. Re-run with -Apply to undeploy these models and stop online endpoint compute charges."
    exit 0
}

if (-not $Force) {
    $answer = Read-Host "Type UNDEPLOY to remove all deployed models from this endpoint"
    if ($answer -ne "UNDEPLOY") {
        Write-Host "Cancelled."
        exit 1
    }
}

foreach ($model in $endpoint.deployedModels) {
    Write-Host ("Undeploying deployed model id={0}..." -f $model.id)
    gcloud ai endpoints undeploy-model $EndpointId `
        --project=$ProjectId `
        --region=$Region `
        --deployed-model-id=$model.id
}

Write-Host "Done. The endpoint can remain as metadata, but it should no longer have deployed model compute running."
