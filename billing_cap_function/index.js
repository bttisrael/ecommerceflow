const functions = require('@google-cloud/functions-framework');
const {CloudBillingClient} = require('@google-cloud/billing');

const projectId = process.env.GOOGLE_CLOUD_PROJECT || process.env.PROJECT_ID;
const projectName = `projects/${projectId}`;
const thresholdRatio = Number(process.env.BILLING_CAP_THRESHOLD_RATIO || '1.0');
const baselineAmount = Number(process.env.BILLING_CAP_BASELINE_AMOUNT || '0');
const dryRun = String(process.env.DRY_RUN || 'false').toLowerCase() === 'true';
const billing = new CloudBillingClient();

function decodeBudgetMessage(cloudEvent) {
  const encoded =
    cloudEvent?.data?.message?.data ||
    cloudEvent?.data?.data ||
    cloudEvent?.data;

  if (!encoded) {
    throw new Error('Pub/Sub event did not include budget message data.');
  }

  return JSON.parse(Buffer.from(encoded, 'base64').toString('utf8'));
}

async function isBillingEnabled() {
  try {
    const [res] = await billing.getProjectBillingInfo({name: projectName});
    return Boolean(res.billingEnabled);
  } catch (err) {
    console.error('Could not read billing status; assuming billing is enabled.', err);
    return true;
  }
}

async function disableBilling() {
  if (dryRun) {
    console.log(`[DRY_RUN] Would disable billing for ${projectName}.`);
    return;
  }

  const [res] = await billing.updateProjectBillingInfo({
    name: projectName,
    resource: {billingAccountName: ''},
  });

  console.log(`Billing disabled for ${projectName}: ${JSON.stringify(res)}`);
}

functions.cloudEvent('stopBilling', async cloudEvent => {
  if (!projectId) {
    throw new Error('GOOGLE_CLOUD_PROJECT or PROJECT_ID is required.');
  }

  const budget = decodeBudgetMessage(cloudEvent);
  const costAmount = Number(budget.costAmount || 0);
  const budgetAmount = Number(budget.budgetAmount || 0);
  const incrementalCost = Math.max(0, costAmount - baselineAmount);
  const stopAt = budgetAmount * thresholdRatio;

  console.log(
    `Budget event: cost=${costAmount}, baseline=${baselineAmount}, incremental=${incrementalCost}, budget=${budgetAmount}, thresholdRatio=${thresholdRatio}, stopAt=${stopAt}`
  );

  if (!budgetAmount || incrementalCost < stopAt) {
    console.log('No billing action needed yet.');
    return;
  }

  if (await isBillingEnabled()) {
    await disableBilling();
  } else {
    console.log('Billing is already disabled.');
  }
});
