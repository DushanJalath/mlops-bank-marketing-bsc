#!/usr/bin/env bash
# Create every Azure resource the workshop needs. Run it in Azure Cloud Shell (Bash):
#
#   ./infra/bootstrap.sh <github-owner>/<repo> <first-image> [region]
#
# Example:
#   ./infra/bootstrap.sh dushan/mlops-bank-marketing ghcr.io/dushan/mlops-bank-marketing:latest
set -euo pipefail

REPO="${1:?usage: bootstrap.sh owner/repo image [region]}"
IMAGE="${2:?usage: bootstrap.sh owner/repo image [region]}"
REGION="${3:-}"
RG="rg-mlops-workshop"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== 1/4 Registering resource providers (first time takes 1-2 minutes)"
for ns in Microsoft.App Microsoft.OperationalInsights Microsoft.Insights Microsoft.ManagedIdentity; do
  az provider register --namespace "$ns" --wait
done

if [ -z "$REGION" ]; then
  echo "== 2/4 Finding a region your subscription allows that offers Container Apps"
  # Azure for Students limits each subscription to a few regions through a policy.
  ALLOWED=$(az policy assignment list -o json | python3 -c '
import json, sys
found = set()
for assignment in json.load(sys.stdin):
    for param in (assignment.get("parameters") or {}).values():
        value = param.get("value") if isinstance(param, dict) else None
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            found.update(x.lower().replace(" ", "") for x in value)
print(" ".join(sorted(found)))')
  ACA=$(az provider show -n Microsoft.App \
    --query "resourceTypes[?resourceType=='containerApps'].locations[]" -o tsv \
    | tr '[:upper:]' '[:lower:]' | tr -d ' ')
  for r in southindia centralindia southeastasia eastasia westindia uaenorth japaneast \
           koreacentral australiaeast northeurope westeurope eastus; do
    if echo "$ACA" | grep -qx "$r" && { [ -z "$ALLOWED" ] || [[ " $ALLOWED " == *" $r "* ]]; }; then
      REGION="$r"
      break
    fi
  done
  if [ -z "$REGION" ]; then
    echo "No suitable region found. Allowed by policy: ${ALLOWED:-unknown}"
    echo "Re-run with a region as the third argument."
    exit 1
  fi
fi
echo "   Using region: $REGION"

# GitHub OIDC tokens use an immutable subject for repositories created after July 15, 2026:
#   repo:<owner>@<owner-id>/<repo>@<repo-id>:...
# Azure must trust exactly that subject, so look the IDs up from the public GitHub API.
echo "   Looking up GitHub IDs for $REPO"
SUBJECT_PREFIX=$(curl -fsSL "https://api.github.com/repos/$REPO" | python3 -c '
import json, sys
d = json.load(sys.stdin)
print("repo:%s@%s/%s@%s" % (d["owner"]["login"], d["owner"]["id"], d["name"], d["id"]))') || {
  echo "Could not read https://api.github.com/repos/$REPO - check the owner/repo name."
  exit 1
}
echo "   Azure will trust: $SUBJECT_PREFIX"

echo "== 3/4 Creating resource group $RG"
az group create --name "$RG" --location "$REGION" --output none

echo "== 4/4 Deploying infra/main.bicep (3-5 minutes)"
OUTPUTS=$(az deployment group create \
  --resource-group "$RG" \
  --name workshop \
  --template-file "$HERE/main.bicep" \
  --parameters githubSubjectPrefix="$SUBJECT_PREFIX" image="$IMAGE" \
  --query properties.outputs --output json)

echo "$OUTPUTS" | python3 -c '
import json, sys
# The Azure CLI can change the case of output names (e.g. "azurE_CLIENT_ID"),
# so look them up case-insensitively.
out = {k.lower(): v["value"] for k, v in json.load(sys.stdin).items()}
print()
print("Done. Add these as GitHub Actions VARIABLES")
print("(repo Settings > Secrets and variables > Actions > Variables tab):")
for key in ["AZURE_CLIENT_ID", "AZURE_TENANT_ID", "AZURE_SUBSCRIPTION_ID",
            "LOG_ANALYTICS_WORKSPACE_ID"]:
    print("  " + key + " = " + out[key.lower()])
print()
print("Your API: " + out["app_url"] + "/health")'
