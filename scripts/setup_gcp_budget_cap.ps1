param(
    [string]$ProjectId = $(if ($env:PROJECT_ID) { $env:PROJECT_ID } else { "otimizador-cargas" }),
    [Parameter(Mandatory = $true)]
    [string]$BillingAccountId,
    [string]$Region = $(if ($env:REGION) { $env:REGION } else { "us-central1" }),
    [string]$BudgetAmount = "30",
    [double]$ThresholdRatio = 1.0,
    [string]$TopicId = "commerceflow-budget-alerts",
    [string]$FunctionName = "commerceflow-stop-billing",
    [string]$BudgetDisplayName = "CommerceFlow hard cap",
    [string]$Runtime = "nodejs22",
    [switch]$Apply
)

$ErrorActionPreference = "Stop"

$gcloudCmd = Get-Command gcloud.cmd -ErrorAction SilentlyContinue
if ($gcloudCmd) {
    $gcloud = $gcloudCmd.Source
} else {
    $gcloudCmd = Get-Command gcloud -ErrorAction SilentlyContinue
    if ($gcloudCmd) {
        $gcloud = $gcloudCmd.Source
    }
}
if (-not $gcloud) {
    throw "gcloud was not found in PATH."
}

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$functionSource = Join-Path $root "billing_cap_function"
$topicName = "projects/$ProjectId/topics/$TopicId"

function Invoke-Step {
    param(
        [string]$Title,
        [string[]]$CommandArgs
    )

    Write-Host ""
    Write-Host "==> $Title"
    Write-Host "$gcloud $($CommandArgs -join ' ')"

    if ($Apply) {
        & $gcloud @CommandArgs
    }
}

Write-Host "Project         : $ProjectId"
Write-Host "Billing account : $BillingAccountId"
Write-Host "Budget amount   : $BudgetAmount"
Write-Host "Threshold ratio : $ThresholdRatio"
Write-Host "Pub/Sub topic   : $topicName"
Write-Host "Function        : $FunctionName"

if (-not $Apply) {
    Write-Host ""
    Write-Host "Dry run only. Re-run with -Apply to create/update resources."
}

Invoke-Step "Enable required APIs" @(
    "services", "enable",
    "billingbudgets.googleapis.com",
    "cloudbilling.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudfunctions.googleapis.com",
    "eventarc.googleapis.com",
    "run.googleapis.com",
    "pubsub.googleapis.com",
    "artifactregistry.googleapis.com",
    "--project=$ProjectId"
)

if ($Apply) {
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $gcloud "pubsub" "topics" "describe" $TopicId "--project=$ProjectId" *> $null
    $topicDescribeExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference

    if ($topicDescribeExitCode -eq 0) {
        Write-Host ""
        Write-Host "==> Pub/Sub topic already exists"
        Write-Host "$topicName"
    } else {
        Invoke-Step "Create Pub/Sub topic" @(
            "pubsub", "topics", "create", $TopicId,
            "--project=$ProjectId"
        )
    }
} else {
    Invoke-Step "Create Pub/Sub topic if needed" @(
        "pubsub", "topics", "create", $TopicId,
        "--project=$ProjectId"
    )
}

Invoke-Step "Deploy billing cap function" @(
    "functions", "deploy", $FunctionName,
    "--gen2",
    "--runtime=$Runtime",
    "--region=$Region",
    "--project=$ProjectId",
    "--source=$functionSource",
    "--entry-point=stopBilling",
    "--trigger-topic=$TopicId",
    "--min-instances=0",
    "--max-instances=1",
    "--memory=256Mi",
    "--timeout=60s",
    "--set-env-vars=GOOGLE_CLOUD_PROJECT=$ProjectId,BILLING_CAP_THRESHOLD_RATIO=$ThresholdRatio,DRY_RUN=false"
)

if ($Apply) {
    $serviceAccount = & $gcloud functions describe $FunctionName `
        "--region=$Region" `
        "--project=$ProjectId" `
        "--format=value(serviceConfig.serviceAccountEmail)"

    if (-not $serviceAccount) {
        throw "Could not resolve deployed function service account."
    }

    Write-Host ""
    Write-Host "Function service account: $serviceAccount"
} else {
    $serviceAccount = "<resolved-after-deploy>"
}

Invoke-Step "Grant function permission to unlink project billing" @(
    "billing", "accounts", "add-iam-policy-binding", $BillingAccountId,
    "--member=serviceAccount:$serviceAccount",
    "--role=roles/billing.admin"
)

Invoke-Step "Create monthly budget wired to Pub/Sub" @(
    "billing", "budgets", "create",
    "--billing-account=$BillingAccountId",
    "--display-name=$BudgetDisplayName",
    "--budget-amount=$BudgetAmount",
    "--calendar-period=month",
    "--filter-projects=projects/$ProjectId",
    "--threshold-rule=percent=$ThresholdRatio",
    "--notifications-rule-pubsub-topic=$topicName"
)

Write-Host ""
Write-Host "Done. When the budget notification crosses the threshold, the function removes Cloud Billing from $ProjectId."
Write-Host "Note: budget notifications can be delayed, so actual spend can exceed the budget before the function runs."
