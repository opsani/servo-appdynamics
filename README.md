# servo-appdynamics
![Run Tests](https://github.com/opsani/servox/workflows/Run%20Tests/badge.svg)
[![license](https://img.shields.io/github/license/opsani/servo-appdynamics.svg)](https://github.com/opsani/servo-appdynamics/blob/master/LICENSE)

Connector for Opsani [Servo](https://github.com/opsani/servox) that utilizes [AppDynamics](https://www.appdynamics.com/) agents to provide metrics for optimization. Either a standard RED (requests-error-duration) measurement set can be used, or a more specific [Business Transactions](https://www.appdynamics.com/product/how-it-works/business-transaction) (BT) set, which are used which map the end-to-end, cross-tier processing path used to fulfill a request for a service provided by the application. For example, in a fully-featured application such as [Bank of Anthos](https://github.com/opsani/bank-of-anthos), the "/payment" BT provides metrics for the full path that BT travels through, including both frontend-service and user-service. Note: due to the AppDynamics mapping process, BT-based optimizations can only be performed through the originating tier of the BT (frontend, in most cases).


## Configuration

```yaml
appdynamics:
  description: Update the app_id, tier, base_url and metrics to match your AppDynamics configuration. 
    Username, account and password set via K8s secrets
  app_id: appd-payment
  tier: frontend-service
  base_url: https://replaceme.saas.appdynamics.com
  metrics:
  - name: main_throughput
    unit: rpm
    query: Overall Application Performance|frontend-service|Individual Nodes|frontend|Calls
      per Minute
```

###

Username, account and password credentials are set in the target kubernetes namespace via k8s secrets, and come applied 
in the servo manifest as follows:

```yaml
- name: SERVO_APPDYNAMICS_USERNAME
  valueFrom:
    secretKeyRef:
      name: appd-secrets
      key: username
- name: SERVO_APPDYNAMICS_ACCOUNT
  valueFrom:
    secretKeyRef:
      name: appd-secrets
      key: accountname
- name: SERVO_APPDYNAMICS_PASSWORD
  valueFrom:
    secretKeyRef:
      name: appd-secrets
      key: password
```

## Usage

Latest image builds are available via `opsani/servox-appdynamics:edge`

Preconfigured metric templates are available via the Opsani console for both RED and BT optimizations.

## Measurements

To differentiate measurements from a main and tuning set, metrics defined in the config are prepended with the respective set name. When measuring a business transaction, the BT is also appended. E.g. `main_request_rate` along with `tuning_request_rate`. Native aggregation occurs within the connector to identify and either average or sum metrics for all active nodes in the main set, and obtained directly in the case of the singleton tuning node. Additive metrics measured in `rpm` such as request rate are aggregated via a sum, and subsequently differentiated in the console to also provide the pod count sensitive value (resulting in `main_request_rate_total` and `main_request_rate`). Metrics measured in `ms` such as latency are returned already averaged to the pod count. 


## License

servo-appdynamics is distributed under the terms of the Apache 2.0 Open Source license.

A copy of the license is provided in the [LICENSE](LICENSE) file at the root of
the repository.
