import pytest
import datetime
import re

import httpx
import respx
import freezegun
import pydantic

import servo
from servo.types import *
from servo_appdynamics import (
    AppdynamicsChecks,
    AppdynamicsConfiguration,
    AppdynamicsMetric,
    AppdynamicsRequest,
    AppdynamicsConnector,
)


class TestAppdynamicsMetric:

    # Query
    def test_query_required(self):
        try:
            AppdynamicsMetric(
                name="throughput", unit=Unit.requests_per_minute, query=None
            )
        except pydantic.ValidationError as error:
            assert {
                "loc": ("query",),
                "msg": "none is not an allowed value",
                "type": "type_error.none.not_allowed",
            } in error.errors()


class TestAppdynamicsConfiguration:
    @pytest.fixture()
    def appdynamics_config(self) -> AppdynamicsConfiguration:
        return AppdynamicsConfiguration(
            metrics=[],
            username="user",
            password="pass",
            account="acc",
            app_id="app",
            tier="test-tier",
        )

    def test_url_required(self, appdynamics_config):
        try:
            appdynamics_config
        except pydantic.ValidationError as error:
            assert {
                "loc": ("base_url",),
                "msg": "none is not an allowed value",
                "type": "type_error.none.not_allowed",
            } in error.errors()

    def test_supports_localhost_url(self):
        config = AppdynamicsConfiguration(
            base_url="http://localhost:8090",
            metrics=[],
            username="user",
            password="pass",
            account="acc",
            app_id="app",
            tier="test-tier",
        )
        assert config.base_url == "http://localhost:8090"

    def test_supports_cluster_url(self):
        config = AppdynamicsConfiguration(
            base_url="http://appdynamics.com:8090",
            metrics=[],
            username="user",
            password="pass",
            account="acc",
            app_id="app",
            tier="test-tier",
        )
        assert config.base_url == "http://appdynamics.com:8090"

    def test_rejects_invalid_url(self):
        try:
            AppdynamicsConfiguration(
                base_url="gopher://this-is-invalid",
                metrics=[],
                username="user",
                password="pass",
                account="acc",
                app_id="app",
                tier="test-tier",
            )
        except pydantic.ValidationError as error:
            assert {
                "loc": ("base_url",),
                "msg": "URL scheme not permitted",
                "type": "value_error.url.scheme",
                "ctx": {
                    "allowed_schemes": {
                        "http",
                        "https",
                    },
                },
            } in error.errors()

    def test_api_url(self):
        config = AppdynamicsConfiguration(
            base_url="http://appdynamics.com:8090",
            metrics=[],
            username="user",
            password="pass",
            account="acc",
            app_id="app",
            tier="test-tier",
        )
        assert config.api_url == "http://appdynamics.com:8090/controller/rest/"

    # Metrics
    def test_metrics_required(self):
        try:
            AppdynamicsConfiguration(
                metrics=None,
                username="user",
                password="pass",
                account="acc",
                app_id="app",
                tier="test-tier",
            )
        except pydantic.ValidationError as error:
            assert {
                "loc": ("metrics",),
                "msg": "none is not an allowed value",
                "type": "type_error.none.not_allowed",
            } in error.errors()


class TestAppdynamicsRequest:
    @freezegun.freeze_time("2020-01-01")
    def test_url(self):
        request = AppdynamicsRequest(
            base_url="http://appdynamics.com",
            start=datetime.datetime.now(),
            end=datetime.datetime.now() + Duration("36h"),
            metric=AppdynamicsMetric(
                "throughput",
                servo.Unit.requests_per_minute,
                query="Overall Application Performance|Calls per Minute",
            ),
        )
        assert (
            request.endpoint
            == "?metric-path=Overall Application Performance|Calls per Minute&time-range-type=BETWEEN_TIMES&start-time=1577836800000&end-time=1577966400000&rollup=false&output=JSON"
        )


node_list_raw = [
    {
        "agentType": "PYTHON_APP_AGENT",
        "appAgentPresent": True,
        "appAgentVersion": "Python Agent v21.2.0.3144 (proxy v21.2.0.31828) "
        "compatible with 4.5.0.21130",
        "id": 64050,
        "ipAddresses": {"ipAddresses": ["192.168.28.154"]},
        "machineAgentPresent": False,
        "machineAgentVersion": "",
        "machineId": 418789,
        "machineName": "5882ee4e6f3d",
        "machineOSType": "Linux",
        "name": "frontend-service--1",
        "nodeUniqueLocalId": "",
        "tierId": 3996,
        "tierName": "frontend-service",
        "type": "Other",
    },
    {
        "agentType": "PYTHON_APP_AGENT",
        "appAgentPresent": True,
        "appAgentVersion": "Python Agent v21.2.0.3144 (proxy v21.2.0.31828) "
        "compatible with 4.5.0.21130",
        "id": 64066,
        "ipAddresses": {"ipAddresses": ["192.168.10.42"]},
        "machineAgentPresent": False,
        "machineAgentVersion": "",
        "machineId": 419058,
        "machineName": "519633c4819c",
        "machineOSType": "Linux",
        "name": "frontend-service--2",
        "nodeUniqueLocalId": "",
        "tierId": 3996,
        "tierName": "frontend-service",
        "type": "Other",
    },
    {
        "agentType": "PYTHON_APP_AGENT",
        "appAgentPresent": True,
        "appAgentVersion": "Python Agent v21.2.0.3144 (proxy v21.2.0.31828) "
        "compatible with 4.5.0.21130",
        "id": 64073,
        "ipAddresses": {"ipAddresses": ["192.168.95.166"]},
        "machineAgentPresent": False,
        "machineAgentVersion": "",
        "machineId": 419111,
        "machineName": "eb15822201ae",
        "machineOSType": "Linux",
        "name": "frontend-tuning--1",
        "nodeUniqueLocalId": "",
        "tierId": 3996,
        "tierName": "frontend-service",
        "type": "Other",
    },
    {
        "agentType": "PYTHON_APP_AGENT",
        "appAgentPresent": True,
        "appAgentVersion": "Python Agent v21.2.0.3144 (proxy v21.2.0.31828) "
        "compatible with 4.5.0.21130",
        "id": 64077,
        "ipAddresses": {"ipAddresses": ["192.168.84.24"]},
        "machineAgentPresent": False,
        "machineAgentVersion": "",
        "machineId": 419160,
        "machineName": "39d42ba4cc85",
        "machineOSType": "Linux",
        "name": "frontend-tuning--2",
        "nodeUniqueLocalId": "",
        "tierId": 3996,
        "tierName": "frontend-service",
        "type": "Other",
    },
    {
        "agentType": "PYTHON_APP_AGENT",
        "appAgentPresent": True,
        "appAgentVersion": "Python Agent v21.2.0.3144 (proxy v21.2.0.31828) "
        "compatible with 4.5.0.21130",
        "id": 64083,
        "ipAddresses": {"ipAddresses": ["192.168.38.84"]},
        "machineAgentPresent": False,
        "machineAgentVersion": "",
        "machineId": 419199,
        "machineName": "330d9a7c7d4f",
        "machineOSType": "Linux",
        "name": "frontend-tuning--3",
        "nodeUniqueLocalId": "",
        "tierId": 3996,
        "tierName": "frontend-service",
        "type": "Other",
    },
    {
        "agentType": "PYTHON_APP_AGENT",
        "appAgentPresent": True,
        "appAgentVersion": "Python Agent v21.2.0.3144 (proxy v21.2.0.31828) "
        "compatible with 4.5.0.21130",
        "id": 64085,
        "ipAddresses": {"ipAddresses": ["192.168.95.128"]},
        "machineAgentPresent": False,
        "machineAgentVersion": "",
        "machineId": 419227,
        "machineName": "2107d4713de6",
        "machineOSType": "Linux",
        "name": "frontend-service--3",
        "nodeUniqueLocalId": "",
        "tierId": 3996,
        "tierName": "frontend-service",
        "type": "Other",
    },
]

overall_application_performance_throughput_raw = [
    {
        "frequency": "ONE_MIN",
        "metricId": 11318904,
        "metricName": "BTM|Application Summary|Calls per Minute",
        "metricPath": "Overall Application Performance|Calls per Minute",
        "metricValues": [
            {
                "count": 5,
                "current": 2098,
                "max": 1119,
                "min": 12,
                "occurrences": 1,
                "standardDeviation": 0,
                "startTimeInMillis": 1614647040000,
                "sum": 2098,
                "useRange": False,
                "value": 2098,
            },
            {
                "count": 5,
                "current": 1249,
                "max": 615,
                "min": 12,
                "occurrences": 1,
                "standardDeviation": 0,
                "startTimeInMillis": 1614647100000,
                "sum": 1249,
                "useRange": False,
                "value": 1249,
            },
            {
                "count": 5,
                "current": 1659,
                "max": 959,
                "min": 12,
                "occurrences": 1,
                "standardDeviation": 0,
                "startTimeInMillis": 1614647160000,
                "sum": 1659,
                "useRange": False,
                "value": 1659,
            },
            {
                "count": 5,
                "current": 1696,
                "max": 861,
                "min": 12,
                "occurrences": 1,
                "standardDeviation": 0,
                "startTimeInMillis": 1614647220000,
                "sum": 1696,
                "useRange": False,
                "value": 1696,
            },
            {
                "count": 5,
                "current": 2549,
                "max": 1315,
                "min": 12,
                "occurrences": 1,
                "standardDeviation": 0,
                "startTimeInMillis": 1614647280000,
                "sum": 2549,
                "useRange": False,
                "value": 2549,
            },
            {
                "count": 5,
                "current": 2474,
                "max": 1380,
                "min": 12,
                "occurrences": 1,
                "standardDeviation": 0,
                "startTimeInMillis": 1614647340000,
                "sum": 2474,
                "useRange": False,
                "value": 2474,
            },
            {
                "count": 5,
                "current": 1603,
                "max": 792,
                "min": 12,
                "occurrences": 1,
                "standardDeviation": 0,
                "startTimeInMillis": 1614647400000,
                "sum": 1603,
                "useRange": False,
                "value": 1603,
            },
            {
                "count": 5,
                "current": 1909,
                "max": 1103,
                "min": 12,
                "occurrences": 1,
                "standardDeviation": 0,
                "startTimeInMillis": 1614647460000,
                "sum": 1909,
                "useRange": False,
                "value": 1909,
            },
            {
                "count": 5,
                "current": 2241,
                "max": 1270,
                "min": 12,
                "occurrences": 1,
                "standardDeviation": 0,
                "startTimeInMillis": 1614647520000,
                "sum": 2241,
                "useRange": False,
                "value": 2241,
            },
            {
                "count": 5,
                "current": 2665,
                "max": 1491,
                "min": 12,
                "occurrences": 1,
                "standardDeviation": 0,
                "startTimeInMillis": 1614647580000,
                "sum": 2665,
                "useRange": False,
                "value": 2665,
            },
        ],
    }
]


class TestAppdynamicsChecks:
    @pytest.fixture
    def metric(self) -> AppdynamicsMetric:
        return AppdynamicsMetric(
            name="throughput",
            unit=Unit.requests_per_minute,
            query="Overall Application Performance|Calls per Minute",
        )

    @pytest.fixture
    def overall_application_performance_throughput(self) -> List[dict]:
        return overall_application_performance_throughput_raw

    @pytest.fixture
    def mocked_api(self, overall_application_performance_throughput):
        with respx.mock(
            base_url="http://localhost:8090", assert_all_called=False
        ) as respx_mock:
            respx_mock.get(re.compile(r"/controller/rest/.+"), name="query").mock(
                httpx.Response(200, json=overall_application_performance_throughput)
            )
            yield respx_mock

    @pytest.fixture
    def checks(self, metric) -> AppdynamicsChecks:
        config = AppdynamicsConfiguration(
            base_url="http://localhost:8090",
            metrics=[metric],
            username="user",
            password="pass",
            account="acc",
            app_id="app",
            tier="test-tier",
        )
        return AppdynamicsChecks(config=config)

    @respx.mock
    @pytest.mark.asyncio
    async def test_check_queries(self, mocked_api, checks) -> None:
        request = mocked_api["query"]
        multichecks = await checks._expand_multichecks()
        check = await multichecks[0]()
        assert request.called
        assert check
        assert (
            check.name
            == r'Run query "Overall Application Performance|Calls per Minute"'
        )
        assert check.id == "check_queries_item_0"
        assert not check.critical
        assert check.success
        assert check.message == "returned 10 results"


class TestAppdynamicsConnector:
    @pytest.fixture
    def metric(self) -> AppdynamicsMetric:
        return AppdynamicsMetric(
            name="test",
            unit=Unit.requests_per_minute,
            query="Overall Application Performance|Calls per Minute",
        )

    @pytest.fixture
    def node_list(self) -> list[dict]:
        return node_list_raw

    @pytest.fixture
    def overall_application_performance_throughput(self) -> list[dict]:
        return overall_application_performance_throughput_raw

    @pytest.fixture
    def mocked_node_list(self, node_list):
        with respx.mock(
            base_url="http://localhost:8090", assert_all_called=False
        ) as respx_mock:
            respx_mock.get(
                "/controller/rest/applications/app-replace/tiers/tier-replace/nodes",
                name="nodes",
                params={"output": "JSON"},
            ).mock(httpx.Response(200, json=node_list))
            yield respx_mock

    @pytest.fixture
    def mocked_metric_data(self, overall_application_performance_throughput):
        with respx.mock(
            base_url="http://localhost:8090", assert_all_called=False
        ) as respx_mock:
            respx_mock.get(
                re.compile(r"/controller/rest/.+"),
                name="query",
            ).mock(httpx.Response(200, json=overall_application_performance_throughput))
            yield respx_mock

    @pytest.fixture
    def connector(self, metric) -> AppdynamicsConnector:
        config = AppdynamicsConfiguration(
            base_url="http://localhost:8090",
            metrics=[metric],
            username="user",
            password="pass",
            account="acc",
            app_id="app-replace",
            tier="tier-replace",
        ).generate(
            username="user",
            password="pass",
            account="acc",
        )
        return AppdynamicsConnector(config=config)

    @pytest.mark.asyncio
    async def test_describe(self, connector) -> None:
        described = connector.describe()
        assert described.metrics == connector.metrics()

    @respx.mock
    @pytest.mark.asyncio
    async def test_query_nodes(self, mocked_node_list, connector) -> None:
        request = mocked_node_list["nodes"]
        nodes = await connector._query_nodes()
        assert request.called
        assert nodes == [node["name"] for node in node_list_raw]

    @respx.mock
    @pytest.mark.asyncio
    async def test_node_response(self, mocked_metric_data, connector) -> None:
        request = mocked_metric_data["query"]
        metrics = connector.metrics()
        start, end = (
            datetime.datetime.now() - datetime.timedelta(minutes=5),
            datetime.datetime.now(),
        )
        measurement = await connector._appd_node_response(
            start=start,
            end=end,
            node="frontend-service--1",
            metric=connector.metrics()[0],
        )
        assert request.called
        assert len(measurement) == 10

        first_datapoint = measurement[0]
        assert type(first_datapoint) == servo.DataPoint
        assert first_datapoint.metric == metrics[0]
        assert first_datapoint.value == 2098.0

    @respx.mock
    @pytest.mark.asyncio
    async def test_instance_count(self, mocked_metric_data, connector) -> None:
        request = mocked_metric_data["query"]
        metrics = connector.metrics()
        start, end = (
            datetime.datetime.now() - datetime.timedelta(minutes=5),
            datetime.datetime.now(),
        )
        active_nodes = [f"frontend-service--{i+1}" for i in range(5)]
        instance_counts = await asyncio.gather(
            connector._query_instance_count(
                start=start,
                end=end,
                active_nodes=active_nodes,
                metric=metrics[0],
            )
        )
        assert request.called
        assert all([i.value == 5.0 for i in instance_counts[0][0].data_points])

    @respx.mock
    @pytest.mark.asyncio
    async def test_readings_transpose(self, mocked_metric_data, connector) -> None:
        request = mocked_metric_data["query"]
        metrics = connector.metrics()
        start, end = (
            datetime.datetime.now() - datetime.timedelta(minutes=5),
            datetime.datetime.now(),
        )
        active_nodes = [f"frontend-service--{i+1}" for i in range(5)]
        node_readings = await asyncio.gather(
            *list(
                map(
                    lambda m: connector._appd_node_response(m, metrics[0], start, end),
                    active_nodes,
                )
            )
        )
        # Raw readings are grouped in [nodes[time]]
        assert all([node[0].time == node_readings[0][0].time for node in node_readings])

        (
            transposed_node_readings,
            max_length_node_items,
        ) = connector.node_sync_and_transpose(node_readings)

        # After transpose, readings are grouped in [times[nodes]]
        for time in transposed_node_readings:
            assert all([node.time == time[0].time for node in time[1:]])

    @respx.mock
    @pytest.mark.asyncio
    async def test_metric_aggregation(self, mocked_metric_data, connector) -> None:
        request = mocked_metric_data["query"]
        metrics = connector.metrics()
        start, end = (
            datetime.datetime.now() - datetime.timedelta(minutes=5),
            datetime.datetime.now(),
        )
        active_nodes = [f"frontend-service--{i+1}" for i in range(5)]
        aggregate_readings = await connector._query_appd_aggregate(
            start=start,
            end=end,
            active_nodes=active_nodes,
            metric=metrics[1],
        )
        assert request.called
        direct_values = [
            metric_value["value"]
            for metric_value in overall_application_performance_throughput_raw[0][
                "metricValues"
            ]
        ]
        for node in aggregate_readings:
            assert all(
                [
                    data_point.value == direct_point
                    for data_point, direct_point in zip(node, direct_values)
                ]
            )
