# servo-appdynamics

[![license](https://img.shields.io/github/license/opsani/servo-appdynamics.svg)](https://github.com/opsani/servo-appdynamics/blob/master/LICENSE)

Connector for Opsani [Servo](https://github.com/opsani/servox) that utilizes [AppDynamics](https://www.appdynamics.com/) agents to provide metrics for optimization. Specifically, [Business Transactions](https://www.appdynamics.com/product/how-it-works/business-transaction) (BTs) are used which map the end-to-end, cross-tier processing path used to fulfill a request for a service provided by the application. For example, in a fully-featured application such as [Bank of Anthos](https://github.com/opsani/bank-of-anthos), the "payment" business transaction provides metrics for the full path that BT travels through, including both frontend-service and user-service. 


## Configuration

```yaml
appdynamics:
  description: Update the app_id, tier, base_url and metrics to match your AppDynamics configuration. 
    Username, account and password set via K8s secrets
  app_id: appd-payment
  tier: frontend-service
  base_url: https://replaceme.saas.appdynamics.com
  metrics:
  - name: main_payment_throughput
    unit: rpm
    query: Business Transaction Performance|Business Transactions|frontend-service|/signup|Individual Nodes|frontend|Calls
      per Minute
```

###

Username, account and password credentials are set in the target kubernetes application via k8s secrets, and applied 
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

## Measurements

To differentiate measurements from a main and tuning set, metrics defined in the config are prepended with the respective set name, as well as the target business transaction. E.g. `main_payment_througphut` along with `tuning_payment_throughput`. Native aggregation occurs within the connector to identify and either average or sum metrics from that BT for all active nodes in the main set, and obtained directly in the case of the singleton tuning node.


## License

servo-appdynamics is distributed under the terms of the Apache 2.0 Open Source license.

A copy of the license is provided in the [LICENSE](LICENSE) file at the root of
the repository.
