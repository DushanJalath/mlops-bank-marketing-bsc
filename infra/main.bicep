// Everything the workshop needs in one resource group.
// Deployed by infra/bootstrap.sh from Azure Cloud Shell.

@description('Azure region. Must be one your Azure for Students subscription allows.')
param location string = resourceGroup().location

@description('GitHub OIDC subject prefix, e.g. repo:owner@123/name@456 (bootstrap.sh builds it).')
param githubSubjectPrefix string

@description('First container image to run, e.g. ghcr.io/owner/mlops-bank-marketing:latest')
param image string

param appName string = 'ca-bank-marketing'

var suffix = uniqueString(resourceGroup().id)

// ---------- identity GitHub Actions uses to log in (no passwords) ----------
resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-github-${suffix}'
  location: location
}

// Trust the main branch (CI, monitoring, retraining) ...
resource ficMain 'Microsoft.ManagedIdentity/userAssignedIdentities/federatedIdentityCredentials@2023-01-31' = {
  parent: identity
  name: 'github-main'
  properties: {
    issuer: 'https://token.actions.githubusercontent.com'
    subject: '${githubSubjectPrefix}:ref:refs/heads/main'
    audiences: ['api://AzureADTokenExchange']
  }
}

// ... and the approved "production" environment (deployments).
// Azure cannot write two credentials on one identity at the same time, hence dependsOn.
resource ficProd 'Microsoft.ManagedIdentity/userAssignedIdentities/federatedIdentityCredentials@2023-01-31' = {
  parent: identity
  name: 'github-production'
  properties: {
    issuer: 'https://token.actions.githubusercontent.com'
    subject: '${githubSubjectPrefix}:environment:production'
    audiences: ['api://AzureADTokenExchange']
  }
  dependsOn: [ficMain]
}

// Contributor on this resource group only.
resource contributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(resourceGroup().id, identity.id, 'contributor')
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      'b24988ac-6180-42a0-ab88-20f7382dd24c'
    )
  }
}

// ---------- logs and metrics ----------
resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-mlops-${suffix}'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource insights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-mlops-${suffix}'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logs.id
  }
}

// ---------- the app ----------
resource env 'Microsoft.App/managedEnvironments@2026-07-01' = {
  name: 'cae-mlops-${suffix}'
  location: location
  properties: {
    // Ask explicitly for a standard (workload profiles) environment. Without this, Azure can
    // now create an "Express" environment, which does not support multiple revisions
    // (canary), revision suffixes or Log Analytics logs.
    environmentMode: 'WorkloadProfiles'
    workloadProfiles: [
      { name: 'Consumption', workloadProfileType: 'Consumption' }
    ]
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  properties: {
    managedEnvironmentId: env.id
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Multiple' // lets old and new versions share traffic (canary)
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        traffic: [
          { latestRevision: true, weight: 100 }
        ]
      }
    }
    template: {
      containers: [
        {
          name: 'api'
          image: image
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: [
            { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: insights.properties.ConnectionString }
          ]
          probes: [
            {
              type: 'Readiness'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 5
              periodSeconds: 10
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 2
        rules: [
          { name: 'http', http: { metadata: { concurrentRequests: '30' } } }
        ]
      }
    }
  }
}

output AZURE_CLIENT_ID string = identity.properties.clientId
output AZURE_TENANT_ID string = tenant().tenantId
output AZURE_SUBSCRIPTION_ID string = subscription().subscriptionId
output LOG_ANALYTICS_WORKSPACE_ID string = logs.properties.customerId
output APP_URL string = 'https://${app.properties.configuration.ingress.fqdn}'
