import asyncio
import datetime
import importlib.metadata
import functools
import re
from typing import Dict, Iterable, List, Optional, Tuple, Union

import httpx
import pydantic

import servo

try:
    __version__ = importlib.metadata.version("servo-appdynamics")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

DEFAULT_BASE_URL = "http://localhost:8090"
API_PATH = "/controller/rest/"


class AppdynamicsMetric(servo.Metric):
    """AppdynamicsMetric objects describe metrics that can be measured by querying
    AppDynamics.
    """

    query: str
    """An AppD query that returns the value of the target metric.

    For details on AppDynamics, see the [AppDynamics
    Querying](https://docs.appdynamics.com/display/PRO45/Metric+and+Snapshot+API)
    documentation.
    """

    @property
    def query_escaped(self) -> str:
        # Not used as such, leaving for now in prom style
        return re.sub(r"\{(.*?)\}", r"{{\1}}", self.query)

    def __check__(self) -> servo.Check:
        return servo.Check(
            name=f"Check {self.name}",
            description=f'Run AppDynamics query "{self.query}"',
        )


class AppdynamicsConfiguration(servo.BaseConfiguration):
    """AppdynamicsConfiguration objects describe how AppdynamicsConnector objects
    capture measurements from the AppDynamics metrics server.
    """

    username: pydantic.SecretStr
    """The username for AppDynamics."""

    account: pydantic.SecretStr
    """The account name for AppDynamics."""

    password: pydantic.SecretStr
    """The API key for accessing the AppDynamics metrics API."""

    app_id: str
    """The Application ID for accessing the AppDynamics metrics API."""

    tier: str
    """The AppDynamics tier that the service to optimize is running on."""

    base_url: pydantic.AnyHttpUrl = DEFAULT_BASE_URL
    """The base URL for accessing the AppDynamics metrics API.

    The URL must point to the root of the AppDynamics deployment. Resource paths
    are computed as necessary for API requests.
    """

    metrics: List[AppdynamicsMetric]
    """The metrics to measure from AppDynamics.

    Metrics must include a valid query.
    """

    @classmethod
    def generate(cls, **kwargs) -> "AppdynamicsConfiguration":
        """Generates a default configuration for capturing measurements from the
        AppDynamics metrics server. As queries are highly app-dependent, they are
        purely for reference and generated to match the AppD Bank of Anthos tutorial.

        Returns:
            A default configuration for AppdynamicsConnector objects.
        """
        return cls(
            description="Update the app_id, tier and base_url and metrics to match your AppDynamics configuration. Username, account and password set via K8s secrets",
            app_id='app-replace',
            tier='tier-replace',
            metrics=[

                # Main metrics
                AppdynamicsMetric(
                    "main_payment_instance_count",
                    servo.Unit.requests_per_minute,
                    query="Business Transaction Performance|Business Transactions|frontend-service|/payment|Individual Nodes|frontend-service|Calls per Minute",
                ),
                AppdynamicsMetric(
                    "main_payment_request_rate_total",
                    servo.Unit.requests_per_minute,
                    query="Business Transaction Performance|Business Transactions|frontend-service|/payment|Individual Nodes|frontend-service|Calls per Minute",
                ),
                AppdynamicsMetric(
                    "main_payment_error_rate_total",
                    servo.Unit.requests_per_minute,
                    query="Business Transaction Performance|Business Transactions|frontend-service|/payment|Individual Nodes|frontend-service|Errors per Minute",
                ),
                AppdynamicsMetric(
                    "main_payment_latency",
                    servo.Unit.milliseconds,
                    query="Business Transaction Performance|Business Transactions|frontend-service|/payment|Individual Nodes|frontend-service|Average Response Time (ms)",
                ),
                AppdynamicsMetric(
                    "main_payment_latency_normal",
                    servo.Unit.milliseconds,
                    query="Business Transaction Performance|Business Transactions|frontend-service|/payment|Individual Nodes|frontend-service|Normal Average Response Time (ms)",
                ),
                AppdynamicsMetric(
                    "main_payment_latency_95th",
                    servo.Unit.milliseconds,
                    query="Business Transaction Performance|Business Transactions|frontend-service|/payment|Individual Nodes|frontend-service|95th Percentile Response Time (ms)",
                ),

                # Tuning instance metrics
                AppdynamicsMetric(
                    "tuning_payment_instance_count",
                    servo.Unit.requests_per_minute,
                    query="Business Transaction Performance|Business Transactions|frontend-service|/payment|Individual Nodes|frontend-service-tuning|Calls per Minute",
                ),
                AppdynamicsMetric(
                    "tuning_payment_request_rate",
                    servo.Unit.requests_per_minute,
                    query="Business Transaction Performance|Business Transactions|frontend-service|/payment|Individual Nodes|frontend-service-tuning|Calls per Minute",
                ),
                AppdynamicsMetric(
                    "tuning_payment_error_rate",
                    servo.Unit.requests_per_minute,
                    query="Business Transaction Performance|Business Transactions|frontend-service|/payment|Individual Nodes|frontend-service-tuning|Errors per Minute",
                ),
                AppdynamicsMetric(
                    "tuning_payment_latency",
                    servo.Unit.milliseconds,
                    query="Business Transaction Performance|Business Transactions|frontend-service|/payment|Individual Nodes|frontend-service-tuning|Average Response Time (ms)",
                ),
                AppdynamicsMetric(
                    "tuning_payment_latency_95th",
                    servo.Unit.milliseconds,
                    query="Business Transaction Performance|Business Transactions|frontend-service|/payment|Individual Nodes|frontend-service-tuning|95th Percentile Response Time (ms)",
                ),
                AppdynamicsMetric(
                    "tuning_payment_latency_normal",
                    servo.Unit.milliseconds,
                    query="Business Transaction Performance|Business Transactions|frontend-service|/payment|Individual Nodes|frontend-service-tuning|Normal Average Response Time (ms)",
                ),
            ],
            **kwargs,
        )

    @property
    def api_url(self) -> str:
        return f"{self.base_url}{API_PATH}"


class AppdynamicsRequest(pydantic.BaseModel):
    base_url: pydantic.AnyHttpUrl
    metric: AppdynamicsMetric
    start: datetime.datetime
    end: datetime.datetime

    @property
    def query(self) -> str:
        return self.metric.query

    @property
    def params(self) -> dict:
        return {"metric-path": f"{self.query}",
                "time-range-type": "BETWEEN_TIMES",  # Should remain non-configurable
                "start-time": f"{int(self.start.timestamp() * 1000)}",
                "end-time": f"{int(self.end.timestamp() * 1000)}",
                "rollup": "false",  # Prevents aggregation/summarization
                "output": "JSON"
                }

    @property
    def endpoint(self) -> str:
        return "".join(f"?metric-path={self.query}"
                       + f"&time-range-type=BETWEEN_TIMES"  # Should remain non-configurable
                       + f"&start-time={int(self.start.timestamp() * 1000)}"
                       + f"&end-time={int(self.end.timestamp() * 1000)}"
                       + f"&rollup=false"  # Prevents aggregation/summarization
                       + f"&output=JSON"
                       )


class AppdynamicsChecks(servo.BaseChecks):
    """AppdynamicsChecks objects check the state of a AppdynamicsConfiguration to
    determine if it is ready for use in an optimization run.
    """

    config: AppdynamicsConfiguration

    @servo.multicheck('Run query "{item.query_escaped}"')
    async def check_queries(self) -> Tuple[Iterable, servo.CheckHandler]:
        """Checks that all metrics have valid, well-formed AppDynamics queries."""

        async def query_for_metric(metric: AppdynamicsMetric) -> str:
            start, end = (
                datetime.datetime.now() - datetime.timedelta(minutes=5),
                datetime.datetime.now(),
            )
            appdynamics_request = AppdynamicsRequest(
                base_url=self.config.api_url, metric=metric, start=start, end=end
            )

            self.logger.trace(
                f"Querying Appdynamics (`{metric.query}`): {appdynamics_request.endpoint}"
            )

            metric_head = '|'.join(metric.query.split('|')[:-3])
            metric_tail = metric.query.split('|')[-1]

            # Ideally we'd check the actual query but there are edge cases where an often "0" metric can mime inactivity
            metric_path_substitution = f"{metric_head}|Calls per Minute"

            params = appdynamics_request.params
            params.update({'metric-path': metric_path_substitution})

            async with httpx.AsyncClient(
                    base_url=self.config.api_url,
                    params=params,
            ) as client:
                try:
                    response = await client.get(f"applications/{self.config.app_id}/metric-data",
                                                auth=(f"{self.config.username.get_secret_value()}@"
                                                      f"{self.config.account.get_secret_value()}",
                                                      self.config.password.get_secret_value()))
                    response.raise_for_status()
                except (httpx.HTTPError, httpx.ReadTimeout, httpx.ConnectError) as error:
                    self.logger.trace(f"HTTP error encountered during GET {appdynamics_request.endpoint}: {error}")
                    raise

            node_data = response.json()

            if not node_data:
                self.logger.trace(f"Metric {metric.query} not returning values")
                raise

            if not node_data[0]["metricValues"]:
                self.logger.trace(f"Metric {metric.query} not returning values")
                raise

            elif node_data[0]["metricValues"]:
                self.logger.trace(f"Verified metric {metric_path_substitution} returning data")
                return f"returned {len(node_data[0]['metricValues'])} results"

        return self.config.metrics, query_for_metric


@servo.metadata(
    description="AppDynamics Connector for Opsani",
    version="0.7.5",
    homepage="https://github.com/opsani/servo-appdynamics",
    license=servo.License.apache2,
    maturity=servo.Maturity.stable,
)
class AppdynamicsConnector(servo.BaseConnector):
    """AppdynamicsConnector objects enable servo assemblies to capture
    measurements from the [AppDynamics](https://appdynamics.io/) metrics server.
    """

    config: AppdynamicsConfiguration

    @servo.on_event()
    async def check(
            self,
            matching: Optional[servo.CheckFilter] = None,
            halt_on: Optional[servo.ErrorSeverity] = servo.ErrorSeverity.critical,
    ) -> List[servo.Check]:
        """Checks that the configuration is valid and the connector can capture
        measurements from AppDynamics.

        Checks are implemented in the AppdynamicsChecks class.

        Args:
            matching (Optional[Filter], optional): A filter for limiting the
                checks that are run. Defaults to None.
            halt_on (Severity, optional): When to halt running checks.
                Defaults to Severity.critical.

        Returns:
            List[Check]: A list of check objects that report the outcomes of the
                checks that were run.
        """
        return await AppdynamicsChecks.run(
            self.config, matching=matching, halt_on=halt_on
        )

    @servo.on_event()
    def describe(self, control: servo.Control = servo.Control()) -> servo.Description:
        """Describes the current state of Metrics measured by querying AppDynamics.

        Returns:
            Description: An object describing the current state of metrics
                queried from AppDynamics.
        """
        return servo.Description(metrics=self.config.metrics)

    @servo.on_event()
    def metrics(self) -> List[AppdynamicsMetric]:
        """Returns the list of Metrics measured through AppDynamics queries.

        Returns:
            List[Metric]: The list of metrics to be queried.
        """
        return self.config.metrics

    @servo.on_event()
    async def measure(
            self, *, metrics: List[str] = None, control: servo.Control = servo.Control()
    ) -> servo.Measurement:
        """Queries AppDynamics for metrics as time series values and returns a
        Measurement object that aggregates the readings for processing by the
        optimizer.

        Args:
            metrics (List[str], optional): A list of the metric names to measure.
                When None, all configured metrics are measured. Defaults to None.
            control (Control, optional): A control descriptor that describes how
                the measurement is to be captured. Defaults to Control().

        Returns:
            Measurement: An object that aggregates the state of the metrics
            queried from AppDynamics.
        """
        if metrics:
            metrics__ = list(filter(lambda m: m.name in metrics, self.metrics()))
        else:
            metrics__ = self.metrics()
        measuring_names = list(map(lambda m: m.name, metrics__))
        self.logger.info(
            f"Starting measurement of {len(metrics__)} metrics: {servo.utilities.join_to_series(measuring_names)}"
        )

        start = datetime.datetime.now() + control.warmup
        end = start + control.duration

        sleep_duration = servo.Duration(control.warmup + control.duration)
        self.logger.info(
            f"Waiting {sleep_duration} during metrics collection ({control.warmup} warmup + {control.duration} duration)..."
        )

        progress = servo.DurationProgress(sleep_duration)

        def notifier(p):
            return self.logger.info(
                p.annotate(f"waiting {sleep_duration} during metrics collection...", False),
                progress=p.progress,
            )

        await progress.watch(notifier)
        self.logger.info(
            f"Done waiting {sleep_duration} for metrics collection, resuming optimization."
        )

        # Separate instance_count metrics away from all_metrics
        all_metrics = list(filter(lambda m: 'instance_count' not in m.name, metrics__))
        instance_metrics = list(filter(lambda m: 'instance_count' in m.name, metrics__))
        main_instance_metric = next(iter(filter(lambda m: 'main' in m.name, instance_metrics)))
        tuning_instance_metric = next(iter(filter(lambda m: 'tuning' in m.name, instance_metrics)))

        # Capture the measurements
        self.logger.info(f"Querying AppDynamics for {len(metrics__)} metrics...")

        # Parse active nodes and metrics between main and tuning
        # Collect nodes
        nodes = await self._query_nodes()

        # Main set
        main_nodes = list(filter(lambda x: 'tuning' not in x, nodes))
        aggregate_metrics = list(filter(lambda m: 'main' in m.name, all_metrics))
        all_main_nodes_response = await asyncio.gather(
            *list(map(lambda m: self._query_appd_node_active(m, aggregate_metrics[0], start, end), main_nodes))
        )
        active_main_nodes = list(filter(lambda x: x is not None, all_main_nodes_response))
        self.logger.info(f"Found {len(active_main_nodes)} active nodes: {active_main_nodes}")

        # Capture measurements that require aggregation
        main_instance_count = await asyncio.gather(
            self._query_instance_count(main_instance_metric, start, end, active_main_nodes))
        aggregate_readings = await asyncio.gather(
            *list(map(lambda m: self._query_appd_aggregate(m, start, end, active_main_nodes), aggregate_metrics))
        )

        # Tuning set
        tuning_nodes = list(filter(lambda x: 'tuning' in x, nodes))
        tuning_metrics = list(filter(lambda m: 'tuning' in m.name, all_metrics))
        tuning_nodes_response = await asyncio.gather(
            *list(map(lambda m: self._query_appd_node_active(m, tuning_metrics[0], start, end), tuning_nodes))
        )
        active_tuning_node = next(iter(filter(lambda x: x is not None, tuning_nodes_response)), None)

        # Capture tuning measurements directly that do not require aggregation/processing
        if not active_tuning_node:

            self.logger.info(
                f"No active tuning node, returning empty readings and will retry before aborting current adjustment")
            tuning_readings = []

        elif active_tuning_node:

            self.logger.info(f"Found active tuning node: {active_tuning_node}")
            tuning_instance_count = await asyncio.gather(
                self._query_instance_count(tuning_instance_metric, start, end, [active_tuning_node]))
            tuning_readings = await asyncio.gather(
                *list(map(lambda m: self._appd_dynamic_node(active_tuning_node, m, start, end), tuning_metrics))
            )

        # Combine and clean
        readings = tuning_readings + aggregate_readings + main_instance_count + tuning_instance_count

        all_readings = (
            functools.reduce(lambda x, y: x + y, readings) if readings else []
        )
        print(all_readings)
        measurement = servo.Measurement(readings=all_readings)
        return measurement

    async def _query_nodes(self) -> List[str]:
        """Queries AppDynamics for a list of all nodes under a given tier (specified in the config), both actively
        reporting and shutdown nodes, to be subsequently filtered for status.

        Returns:
            nodes: A list of all nodes within the specified tier.
        """

        self.logger.trace(
            f"Querying AppDynamics nodes for tier: {self.config.tier}"
        )

        endpoint = "/tiers/{self.config.tier}/nodes"
        data = self._appd_api(endpoint=endpoint)
        nodes = [node['name'] for node in data]

        self.logger.trace(f"Retrieved nodes for tier {self.config.tier}: {nodes}")
        return nodes

    async def _query_appd_node_active(
            self, individual_node: str, metric: AppdynamicsMetric, start: datetime, end: datetime
    ) -> Optional[str]:
        """Queries AppDynamics to see if a given node is actively reporting.

        Args:
            individual_node (str, required): The node to query.
            metric (AppdynamicsMetric, required): The metric to query for.
            start (datetime, required): Metric start time.
            end (datetime, required). Metric end time.

        Returns:
            Optional[str]. The node name if actively reporting, otherwise None.
        """

        appdynamics_request = AppdynamicsRequest(
            base_url=self.config.api_url, metric=metric, start=start, end=end
        )

        metric_head = '|'.join(metric.query.split('|')[:-2])
        metric_tail = metric.query.split('|')[
            -1]  # TODO: rationalize having this available - throughput (below) should always suffice
        metric_path_substitution = f"{metric_head}|{individual_node}|Calls per Minute"

        self.logger.trace(
            f"Querying AppDynamics (`{metric_path_substitution}`): {appdynamics_request.endpoint}"
        )

        params = appdynamics_request.params
        params.update({'metric-path': metric_path_substitution})

        node_data = self._appd_api(params=params)

        if not node_data:
            self.logger.trace(f"Found inactive node: {individual_node}")
            return None

        if not node_data[0]["metricValues"]:

            self.logger.trace(f"Found inactive node: {individual_node}")
            return None

        elif node_data[0]["metricValues"]:

            self.logger.trace(f"Verified active node: {individual_node}")
            return individual_node

    async def _query_appd_aggregate(
            self, metric: AppdynamicsMetric, start: datetime, end: datetime, active_nodes: List[str]
    ):
        """Queries AppDynamics for measurements that need to be aggregated across multiple AppD nodes/K8s pods.
        Individual node responses are gathered via _appd_node_response(), transposed to synchronize reading times,
        and computed via either sum or average.

        Args:
            metric (AppdynamicsMetric, required): The metric to query for.
            start (datetime, required): Metric start time.
            end (datetime, required). Metric end time.
            active_nodes (List[str], required): The list of actively reporting nodes to aggregate on

        Returns:
            Readings: A list of TimeSeries with metric readings.
        """

        # Begin metric collection and aggregation for active nodes
        node_readings = await asyncio.gather(
            *list(map(lambda m: self._appd_node_response(m, metric, start, end), active_nodes))
        )

        aggregate_readings = []

        # Transpose node readings from [nodes[readings]] to [readings[nodes]] for computed aggregation
        transposed_node_readings, max_length_node_items = node_sync_and_transpose(node_readings)

        for node_readings in transposed_node_readings:
            aggregate_data_points: List[Union[int, float]] = []

            for data_points in node_readings:
                aggregate_data_points.append(data_points.value)

            # Main aggregation logic
            if metric.unit == 'rpm':
                computed_aggregate = sum(aggregate_data_points)

                self.logger.trace(
                    f"Aggregating values {aggregate_data_points} for {metric.unit} via sum into {computed_aggregate}")
                aggregate_readings.append(computed_aggregate)

            elif metric.unit == 'ms':
                computed_aggregate = sum(aggregate_data_points) / len(aggregate_data_points)

                self.logger.trace(
                    f"Aggregating values {aggregate_data_points} for {metric.unit} via average into {computed_aggregate}")
                aggregate_readings.append(computed_aggregate)

        # Syncing aggregate values to data_points from max_length node readings
        # appdynamics_request = AppdynamicsRequest(
        #     base_url=self.config.api_url, metric=metric, start=start, end=end
        # )
        #
        # self.logger.trace(
        #     f"Querying AppDynamics (`{metric.query}`): {appdynamics_request.endpoint}"
        # )
        #
        # metric_head = '|'.join(metric.query.split('|')[:-2])
        # metric_tail = metric.query.split('|')[-1]
        # metric_path_substitution = f"{metric_head}|{active_nodes[0]}|{metric_tail}"
        #
        # params = appdynamics_request.params
        # params.update({'metric-path': metric_path_substitution})
        #
        # data = self.appd_api(params=params)
        #
        # readings = []
        #
        # for result_dict in zip(aggregate_readings, data["metricValues"]):
        #     self.logger.trace(
        #         f"Captured {result_dict[0]} at {result_dict[1]['startTimeInMillis']} for aggregate metric: {metric}"
        #     )
        #
        # metric_path = data[
        #     "metricPath"]  # e.g. "Business Transaction Performance|Business Transactions|frontend-service|/payment|Calls per Minute"
        # metric_name = data["metricName"]  # e.g. "BTM|BTs|BT:270723|Component:8435|Calls per Minute"
        #
        # data_points: List[servo.DataPoint] = []
        #
        # for result_dict in zip(aggregate_readings, data["metricValues"]):
        #     data_points.append(servo.DataPoint(
        #         metric, result_dict[1]['startTimeInMillis'], float(result_dict[0])
        #     ))
        #
        # readings.append(
        #     servo.TimeSeries(
        #         metric,
        #         data_points,
        #         id=f"{{metric_path={metric_path}, metric_name={metric_name}}}",
        #     )
        # )

        readings = []
        data_points: List[servo.DataPoint] = []

        for max_items, aggregate_value in zip(max_length_node_items, aggregate_readings):
            self.logger.trace(
                f"Syncing aggregate metric {metric.name} value {aggregate_value} to time {max_items.time}"
            )
            data_points.append(servo.DataPoint(
                metric, max_items.time, float(aggregate_value)
            ))

        readings.append(
            servo.TimeSeries(
                metric,
                data_points,
            )
        )

        return readings

    async def _appd_node_response(
            self, individual_node: str, metric: AppdynamicsMetric, start: datetime, end: datetime
    ):
        """Queries AppDynamics for measurements either used directly or in aggregation when a dynamic node is being read
        that requires the metric endpoint to be substituted from the config. Substitutes the metric path with the
        individual nodes' endpoint, as well as substitutes reading values of 0 when the response is empty from a metric
        that does always report (calls per minute) to synchronize timestamps and number of readings.

        Args:
            individual_node (str, required): The active node to be queried.
            metric (AppdynamicsMetric, required): The metric to query for.
            start (datetime, required): Metric start time.
            end (datetime, required). Metric end time.

        Returns:
            Readings: A list of TimeSeries with metric readings.
        """

        appdynamics_request = AppdynamicsRequest(
            base_url=self.config.api_url, metric=metric, start=start, end=end
        )

        metric_head = '|'.join(metric.query.split('|')[:-2])
        metric_tail = metric.query.split('|')[-1]
        metric_path_substitution = f"{metric_head}|{individual_node}|{metric_tail}"

        self.logger.trace(
            f"Querying AppDynamics (`{metric_path_substitution}`): {appdynamics_request.endpoint}"
        )

        params = appdynamics_request.params
        params.update({'metric-path': metric_path_substitution})

        node_data = self._appd_api(params=params)
        self.logger.trace(f"Got response data for metric {metric_path_substitution}: {node_data}")

        data_points: List[servo.DataPoint] = []

        # If the metric data isn't consistently above 0, sometimes no data is returned
        # This requires a substitution with a working call to sync timestamps and number of readings
        # This function is only called for just-verified active nodes

        if not node_data["metricValues"]:
            self.logger.trace(
                f"Metric {metric_tail} failed for individual node: {individual_node}, substituting for Calls per Minute"
            )

            # Calls per Minute is an always-reporting metric that is good to substitute
            metric_path_substitution = f"{metric_head}|{individual_node}|Calls per Minute"

            params = appdynamics_request.params
            params.update({'metric-path': metric_path_substitution})

            node_data = self._appd_api(params=params)
            self.logger.trace(f"Got substitute data for {metric_tail} on node: {individual_node}")

            # Substitute in 0's for the actual metric values
            for result_dict in node_data["metricValues"]:
                data_point = servo.DataPoint(
                    metric, result_dict['startTimeInMillis'], float(0)
                )
                data_points.append(data_point)

        # Main capture logic
        for result_dict in node_data["metricValues"]:
            self.logger.trace(
                f"Captured {result_dict['value']} at {result_dict['startTimeInMillis']} for {metric_path_substitution}"
            )
            data_point = servo.DataPoint(
                metric, result_dict['startTimeInMillis'], float(0)
            )
            data_points.append(data_point)

        return data_points

    async def _query_instance_count(
            self, metric: AppdynamicsMetric, start: datetime, end: datetime, active_nodes: List[str]
    ):
        """Queries AppDynamics for instances count. Individual node responses are gathered via _appd_node_response(),
        transposed to synchronize reading times, and computed via either sum or average.

        Args:
            metric (AppdynamicsMetric, required): The metric to query for.
            start (datetime, required): Metric start time.
            end (datetime, required). Metric end time.
            active_nodes (List[str], required): The list of actively reporting nodes to aggregate on

        Returns:
            Readings: A list of TimeSeries with metric readings.
        """

        # Begin metric collection and aggregation for active nodes

        readings: List[servo.TimeSeries] = []
        data_points: List[servo.DataPoint] = []

        if len(active_nodes) > 1:
            node_readings = await asyncio.gather(
                *list(map(lambda m: self._appd_node_response(m, metric, start, end), active_nodes))
            )
            instance_count_readings: List[int] = []

            # Transpose node readings from [nodes[readings]] to [readings[nodes]] for computed aggregation
            transposed_node_readings, max_length_node_items = node_sync_and_transpose(node_readings)

            for node_reading in transposed_node_readings:
                value = [reading.value for reading in node_reading]
                nonzero_instance_count = list(filter(lambda x: x > 0, value))
                instant_instance_count = len(nonzero_instance_count)
                instance_count_readings.append(instant_instance_count)

                self.logger.trace(
                    f"Found instance count {instant_instance_count} at time {node_reading[0].time}")

            for max_items, aggregate_value in zip(max_length_node_items, instance_count_readings):
                self.logger.trace(
                    f"Syncing aggregate metric {metric.name} value {aggregate_value} to time {max_items.time}"
                )
                data_points.append(servo.DataPoint(
                    metric, max_items.time, float(aggregate_value)
                ))

        # TODO: Cleanup this conditional handling for single instance counting
        elif len(active_nodes) == 1:

            node_readings = await asyncio.gather(self._appd_node_response(active_nodes[0], metric, start, end))
            instance_count_readings = [float(1) for reading in node_readings]

            for reading in node_readings:
                self.logger.trace(
                    f"Found single instance count at time {reading[0].time}")

            for item, instance_value in zip(node_readings, instance_count_readings):
                self.logger.trace(
                    f"Syncing instance count {instance_value} to time {item.time}"
                )
                data_points.append(servo.DataPoint(
                    metric, item.time, float(instance_value)
                ))

        readings.append(
            servo.TimeSeries(
                metric,
                data_points,
            )
        )

        return readings

    async def _appd_dynamic_node(
            self, individual_node: str, metric: AppdynamicsMetric, start: datetime, end: datetime
    ):
        """Queries AppDynamics for measurements used directly when a dynamic node is being read that requires the
        metric endpoint to be substituted from the config. Substitutes the metric path with the individual nodes
        endpoint, as well as substitutes reading values of 0 when the response is empty from a metric that does always
        report (calls per minute) to synchronize timestamps and number of readings.

        Args:
            individual_node (str, required): The active node to be queried.
            metric (AppdynamicsMetric, required): The metric to query for.
            start (datetime, required): Metric start time.
            end (datetime, required). Metric end time.

        Returns:
            Readings: A list of TimeSeries with metric readings.
        """

        appdynamics_request = AppdynamicsRequest(
            base_url=self.config.api_url, metric=metric, start=start, end=end
        )

        metric_head = '|'.join(metric.query.split('|')[:-2])
        metric_tail = metric.query.split('|')[-1]
        metric_path_substitution = f"{metric_head}|{individual_node}|{metric_tail}"

        self.logger.trace(
            f"Querying AppDynamics (`{metric_path_substitution}`): {appdynamics_request.endpoint}"
        )

        params = appdynamics_request.params
        params.update({'metric-path': metric_path_substitution})

        node_data = self._appd_api(params=params)
        self.logger.trace(f"Got response data for metric {metric_path_substitution}: {node_data}")

        metric_path = node_data[
            "metricPath"]  # e.g. "Business Transaction Performance|Business Transactions|frontend-service|/payment|Calls per Minute"
        metric_name = node_data["metricName"]  # e.g. "BTM|BTs|BT:270723|Component:8435|Calls per Minute"

        node_readings: List[servo.TimeSeries] = []
        data_points: List[servo.DataPoint] = []

        # If the metric data isn't consistently above 0, sometimes no data is returned
        # This requires a substitution with a working call to sync timestamps and number of readings
        # This function is only called for just-verified active nodes

        if not node_data["metricValues"]:
            # Calls per Minute is an always-reporting metric that is good to substitute
            self.logger.trace(
                f"Metric {metric_tail} failed for individual node: {individual_node}, substituting for Calls per Minute"
            )

            metric_path_substitution = f"{metric_head}|{individual_node}|Calls per Minute"
            params = appdynamics_request.params
            params.update({'metric-path': metric_path_substitution})

            node_data = self._appd_api(params=params)
            self.logger.trace(f"Got substitute data for {metric_tail} on node: {individual_node}")

            # Substitute in 0's for the actual metric values
            for result_dict in node_data["metricValues"]:
                data_points.append(servo.DataPoint(
                    metric, result_dict['startTimeInMillis'], float(0)
                ))

        # Main capture logic
        for result_dict in node_data["metricValues"]:
            self.logger.trace(
                f"Captured {result_dict['value']} at {result_dict['startTimeInMillis']} for {metric_path_substitution}"
            )
            data_points.append(servo.DataPoint(
                metric, result_dict['startTimeInMillis'], float(result_dict['value'])
            ))

        node_readings.append(
            servo.TimeSeries(
                metric,
                data_points,
                id=f"{{metric_path={metric_path}, metric_name={metric_name}}}",

            )
        )

        return node_readings

    async def _query_appd_direct(
            self, metric: AppdynamicsMetric, start: datetime, end: datetime
    ) -> List[servo.TimeSeries]:
        """Queries AppDynamics for measurements that are taken from a metric exactly as defined in the config, e.g.
        from nodes that are not dynamic and remain consistent. Currently not utilized in the main/tuning workflow as
        both of these utilize a dynamic node naming.

        Args:
            metric (AppdynamicsMetric, required): The metric to query for.
            start (datetime, required): Metric start time.
            end (datetime, required). Metric end time.

        Returns:
            Readings: A list of TimeSeries with metric readings.
        """

        appdynamics_request = AppdynamicsRequest(
            base_url=self.config.api_url, metric=metric, start=start, end=end
        )

        self.logger.trace(
            f"Querying AppDynamics (`{metric.query}`): {appdynamics_request.endpoint}"
        )

        data = self._appd_api(params=appdynamics_request.params)
        self.logger.trace(f"Got response data for metric {metric}: {data}")

        readings: List[servo.TimeSeries] = []

        for result_dict in data["metricValues"]:
            self.logger.trace(
                f"Captured {result_dict['value']} at {result_dict['startTimeInMillis']} for {metric}"
            )

        metric_path = data[
            "metricPath"]  # e.g. "Business Transaction Performance|Business Transactions|frontend-service|/payment|Individual Nodes|frontend-tuning|Calls per Minute"
        metric_name = data["metricName"]  # e.g. "BTM|BTs|BT:270723|Component:8435|Calls per Minute"

        data_points: List[servo.DataPoint] = []

        for result_dict in data["metricValues"]:
            data_points.append(servo.DataPoint(
                metric, result_dict['startTimeInMillis'], float(result_dict['value'])
            ))

        readings.append(
            servo.TimeSeries(
                metric,
                data_points,
                id=f"{{metric_path={metric_path}, metric_name={metric_name}}}",
            )
        )

        return readings

    async def _appd_api(self, endpoint: str = "/metric-data", params: dict = {'output': 'JSON'}) -> list:
        """Base function for accessing the AppDynamics API. Reads credentials from the config loaded by the connector

        Args:
            endpoint (str, optional): Endpoint to access (currently /nodes is the only non-default value utilized).
            params (dict, optional): Params to pass, required for all metric-data calls.
        Returns:
            data: Raw data from the API.
        """

        async with httpx.AsyncClient(
                base_url=self.config.api_url,
                params=params,
        ) as client:
            try:
                response = await client.get(f"applications/{self.config.app_id}/{endpoint}",
                                            auth=(f"{self.config.username.get_secret_value()}@"
                                                  f"{self.config.account.get_secret_value()}",
                                                  self.config.password.get_secret_value()))
                response.raise_for_status()
            except (httpx.HTTPError, httpx.ReadTimeout, httpx.ConnectError) as error:
                self.logger.trace(f"HTTP error encountered during GET {response.url}: {error}")
                raise

        data = response.json()[0]
        return data


async def node_sync_and_transpose(
        node_readings: List[List[servo.DataPoint]]
) -> Tuple[List[List[servo.DataPoint]], List[servo.DataPoint]]:
    """Converts a multi-node aggregate response into a uniform reading, inserting substitute zero value metric data
    for times the node (thus pod) did not exist, synchronized to timestamps from the longest living node within the
    measurement cycle. Transposes the nested list from [nodes[readings]] to [readings[nodes]] for operations, and
    additionally returns all data points from the longest lived node to prevent re-querying.

    Args:
        node_readings (List[List[servo.DataPoint]], required): Nested list of node readings.
    Returns:
        (transposed_node_readings, max_length_node_items): A tuple of the synced+converted readings with the
        readings from the longest-lived node.
    """
    readings_lengths = [len(node) for node in node_readings]
    max_length = max(readings_lengths)
    max_length_node_index, max_length_node_items = [(index, items) for index, items in enumerate(node_readings) if len(items) == max_length][0]
    max_length_times = [reading.time for reading in max_length_node_items]

    # Pad 0 values for nodes with shorter lives, synced to timestamp of longest-lived node
    for node in node_readings:
        times = [reading.time for reading in node]
        unset_readings = list(set(max_length_times) - set(times))
        for time in unset_readings:
            datapoint = servo.DataPoint(
                node[0].metric, time, float(0)
            )
            node.append(datapoint)
        node.sort(key=lambda x: x.time)

    transposed_node_readings = list(map(list, zip(*node_readings)))
    return transposed_node_readings, max_length_node_items
