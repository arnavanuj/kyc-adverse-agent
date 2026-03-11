from __future__ import annotations

import os
import time as time_module
from collections.abc import Iterable
from datetime import date, datetime, time
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
OPENAPI_URL = f"{API_BASE_URL}/openapi.json"
REQUEST_TIMEOUT_SECONDS = 180
HTTP_METHODS = ["get", "post", "put", "patch", "delete"]


@st.cache_data(ttl=30, show_spinner=False)
def fetch_openapi_schema() -> dict[str, Any]:
    response = requests.get(OPENAPI_URL, timeout=10)
    response.raise_for_status()
    return response.json()


def resolve_schema_references(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    if "$ref" in schema:
        ref = schema["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/"):
            return {}
        node: Any = root
        for part in ref.lstrip("#/").split("/"):
            node = node.get(part, {})
        return resolve_schema_references(node, root) if isinstance(node, dict) else {}

    resolved = dict(schema)
    if "properties" in resolved and isinstance(resolved["properties"], dict):
        props: dict[str, Any] = {}
        for key, value in resolved["properties"].items():
            props[key] = resolve_schema_references(value, root) if isinstance(value, dict) else value
        resolved["properties"] = props
    if "items" in resolved and isinstance(resolved["items"], dict):
        resolved["items"] = resolve_schema_references(resolved["items"], root)
    if "anyOf" in resolved and isinstance(resolved["anyOf"], list):
        resolved["anyOf"] = [
            resolve_schema_references(item, root) if isinstance(item, dict) else item
            for item in resolved["anyOf"]
        ]
    return resolved


def normalize_schema(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    resolved = resolve_schema_references(schema, root)
    if "anyOf" in resolved:
        options = [item for item in resolved["anyOf"] if isinstance(item, dict)]
        non_null = [item for item in options if item.get("type") != "null"]
        if len(non_null) == 1:
            merged = dict(non_null[0])
            merged["nullable"] = True
            return merged
    return resolved


def schema_type(schema: dict[str, Any]) -> str:
    type_name = schema.get("type")
    return type_name if isinstance(type_name, str) else "string"


def to_iso_datetime(selected_date: date, selected_time: time) -> str:
    return datetime.combine(selected_date, selected_time).isoformat()


def parse_array_input(raw_value: str, item_schema: dict[str, Any]) -> list[Any]:
    item_type = schema_type(item_schema)
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    if item_type == "integer":
        return [int(v) for v in values]
    if item_type == "number":
        return [float(v) for v in values]
    if item_type == "boolean":
        return [v.lower() in {"true", "1", "yes", "y"} for v in values]
    return values


def nested_assign(target: dict[str, Any], path: str, value: Any) -> None:
    if not path:
        return
    parts = path.split(".")
    current = target
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def render_primitive_input(
    key_prefix: str,
    field_name: str,
    field_schema: dict[str, Any],
    is_required: bool,
) -> Any:
    label = field_name if is_required else f"{field_name} (optional)"
    input_key = f"{key_prefix}.{field_name}"
    field_type = schema_type(field_schema)
    field_format = field_schema.get("format")
    default_value = field_schema.get("default")

    if not is_required:
        include_value = st.checkbox(f"Set {label}", value=False, key=f"{input_key}.enabled")
        if not include_value:
            return None

    if field_format == "date":
        selected = st.date_input(label, key=input_key)
        return selected.isoformat()

    if field_format == "date-time":
        col1, col2 = st.columns(2)
        selected_date = col1.date_input(f"{label} - date", key=f"{input_key}.date")
        selected_time = col2.time_input(f"{label} - time", key=f"{input_key}.time")
        return to_iso_datetime(selected_date, selected_time)

    if field_type == "boolean":
        return st.checkbox(label, value=bool(default_value) if default_value is not None else False, key=input_key)

    if field_type == "integer":
        value = st.number_input(
            label,
            step=1,
            value=int(default_value) if isinstance(default_value, int) else 0,
            key=input_key,
        )
        return int(value)

    if field_type == "number":
        value = st.number_input(
            label,
            value=float(default_value) if isinstance(default_value, (int, float)) else 0.0,
            key=input_key,
        )
        return float(value)

    if isinstance(default_value, str) and default_value:
        return st.text_input(label, value=default_value, key=input_key)
    return st.text_input(label, key=input_key)


def render_schema_fields(
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    key_prefix: str,
    required_fields: set[str] | None = None,
) -> dict[str, Any]:
    required_fields = required_fields or set()
    values: dict[str, Any] = {}

    normalized = normalize_schema(schema, root_schema)
    if schema_type(normalized) != "object":
        return values

    for field_name, field_schema_raw in normalized.get("properties", {}).items():
        if not isinstance(field_schema_raw, dict):
            continue

        field_schema = normalize_schema(field_schema_raw, root_schema)
        required = field_name in required_fields
        field_path = f"{key_prefix}.{field_name}" if key_prefix else field_name
        field_type = schema_type(field_schema)

        if field_type == "object":
            st.markdown(f"**{field_name}**")
            nested_required = set(field_schema.get("required", []))
            nested_values = render_schema_fields(field_schema, root_schema, field_path, nested_required)
            values.update(nested_values)
            continue

        if field_type == "array":
            label = field_name if required else f"{field_name} (optional)"
            raw_value = st.text_input(
                f"{label} (comma-separated)",
                key=f"{field_path}.array",
            )
            if raw_value.strip():
                item_schema_raw = field_schema.get("items", {})
                item_schema = normalize_schema(item_schema_raw, root_schema) if isinstance(item_schema_raw, dict) else {}
                try:
                    parsed = parse_array_input(raw_value, item_schema)
                    values[field_path] = parsed
                except ValueError:
                    st.error(f"Invalid list input for '{field_name}'.")
            continue

        primitive_value = render_primitive_input(key_prefix, field_name, field_schema, required)
        if required or primitive_value not in ("", None):
            values[field_path] = primitive_value

    return values


def endpoint_options(openapi_schema: dict[str, Any]) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for path, methods in openapi_schema.get("paths", {}).items():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if method.lower() not in HTTP_METHODS or not isinstance(operation, dict):
                continue
            summary = operation.get("summary", "").strip()
            label = f"{method.upper()} {path}" + (f" - {summary}" if summary else "")
            options.append(
                {
                    "label": label,
                    "method": method.upper(),
                    "path": path,
                    "operation": operation,
                }
            )
    return sorted(options, key=lambda item: (item["path"], item["method"]))


def split_parameters(operation: dict[str, Any], root_schema: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {"path": [], "query": [], "header": [], "cookie": []}
    parameters = operation.get("parameters", [])
    if not isinstance(parameters, list):
        return grouped

    for parameter in parameters:
        if not isinstance(parameter, dict):
            continue
        resolved = normalize_schema(parameter, root_schema)
        location = resolved.get("in")
        if location in grouped:
            grouped[location].append(resolved)
    return grouped


def render_parameter_inputs(
    parameters: Iterable[dict[str, Any]],
    location: str,
    root_schema: dict[str, Any],
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for parameter in parameters:
        name = parameter.get("name")
        if not isinstance(name, str):
            continue
        required = bool(parameter.get("required", False))
        param_schema_raw = parameter.get("schema", {})
        param_schema = normalize_schema(param_schema_raw, root_schema) if isinstance(param_schema_raw, dict) else {}
        val = render_primitive_input(f"{location}.{name}", name, param_schema, required)
        if required or val not in ("", None):
            values[name] = val
    return values


def build_request_payload(
    endpoint: dict[str, Any],
    root_schema: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, str], dict[str, Any]]:
    operation = endpoint["operation"]
    parameter_groups = split_parameters(operation, root_schema)

    path_values = render_parameter_inputs(parameter_groups["path"], "path", root_schema)
    query_values = render_parameter_inputs(parameter_groups["query"], "query", root_schema)
    header_values = render_parameter_inputs(parameter_groups["header"], "header", root_schema)

    body_payload: dict[str, Any] = {}
    request_body = operation.get("requestBody", {})
    if isinstance(request_body, dict):
        content = request_body.get("content", {})
        if isinstance(content, dict) and "application/json" in content:
            media = content["application/json"]
            if isinstance(media, dict):
                body_schema_raw = media.get("schema", {})
                if isinstance(body_schema_raw, dict):
                    body_schema = normalize_schema(body_schema_raw, root_schema)
                    required_fields = set(body_schema.get("required", []))
                    st.markdown("### Request Body")
                    flat_values = render_schema_fields(body_schema, root_schema, "body", required_fields)
                    for field_path, value in flat_values.items():
                        nested_key = field_path.removeprefix("body.")
                        nested_assign(body_payload, nested_key, value)

    formatted_path = endpoint["path"]
    for key, value in path_values.items():
        formatted_path = formatted_path.replace(f"{{{key}}}", quote(str(value), safe=""))

    return formatted_path, query_values, header_values, {}, body_payload


def render_list(items: list[Any], title: str | None = None) -> None:
    if title:
        st.markdown(f"### {title}")
    if not items:
        st.write("No items.")
        return
    if all(isinstance(item, dict) for item in items):
        df = pd.DataFrame(items)
        for col in ("risk_score", "confidence"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").round(2)
        if not df.empty:
            df.index = df.index + 1
        st.table(df)
        return
    for item in items:
        st.markdown(f"- {item}")


def render_section(key: str, value: Any) -> None:
    title = key.replace("_", " ").title()
    if isinstance(value, dict):
        st.markdown(f"### {title}")
        render_response_object(value)
        return
    if isinstance(value, list):
        render_list(value, title=title)
        return
    st.markdown(f"### {title}")
    st.write(value)


def render_response_object(response_data: dict[str, Any]) -> None:
    special_keys = {"summary", "overall_risk", "overall_score", "key_findings", "recommendations", "metadata", "report"}

    if "summary" in response_data:
        st.markdown("### Summary")
        st.write(response_data["summary"])

    if "overall_risk" in response_data or "overall_score" in response_data:
        st.markdown("### Risk Level")
        col1, col2 = st.columns(2)
        col1.metric("Overall Risk", str(response_data.get("overall_risk", "N/A")).upper())
        score = response_data.get("overall_score")
        col2.metric("Overall Score", f"{score:.3f}" if isinstance(score, (int, float)) else "N/A")

    if isinstance(response_data.get("key_findings"), list):
        render_list(response_data.get("key_findings", []), title="Key Findings")

    if isinstance(response_data.get("recommendations"), list):
        render_list(response_data.get("recommendations", []), title="Recommendations")

    if isinstance(response_data.get("metadata"), dict):
        st.markdown("### Metadata")
        st.table([{"field": k, "value": v} for k, v in response_data["metadata"].items()])

    if isinstance(response_data.get("report"), dict):
        st.markdown("### Report")
        render_response_object(response_data["report"])

    for key, value in response_data.items():
        if key in special_keys:
            continue
        render_section(key, value)


def render_response_friendly(response_data: Any) -> None:
    st.subheader("Response")
    if isinstance(response_data, dict):
        render_response_object(response_data)
        return
    if isinstance(response_data, list):
        render_list(response_data)
        return
    st.write(response_data)


def perform_request(
    method: str,
    path: str,
    query_params: dict[str, Any],
    header_params: dict[str, Any],
    body_payload: dict[str, Any],
) -> requests.Response:
    url = f"{API_BASE_URL}{path}"
    headers = {k: str(v) for k, v in header_params.items()}
    timeout_value: float | None = REQUEST_TIMEOUT_SECONDS
    if method == "POST" and path == "/screening/run":
        timeout_value = None
    kwargs: dict[str, Any] = {
        "params": query_params or None,
        "headers": headers or None,
        "timeout": timeout_value,
    }
    if method in {"POST", "PUT", "PATCH"}:
        kwargs["json"] = body_payload or {}
    return requests.request(method, url, **kwargs)


def main() -> None:
    st.set_page_config(page_title="KYC Adverse Media Screening System", layout="wide")
    st.title("KYC Adverse Media Screening System")
    st.caption(f"Backend API: {API_BASE_URL}")

    try:
        with st.spinner("Loading API schema..."):
            openapi_schema = fetch_openapi_schema()
    except requests.RequestException as exc:
        st.error(f"Unable to load OpenAPI schema from {OPENAPI_URL}. Error: {exc}")
        st.stop()

    endpoints = endpoint_options(openapi_schema)
    if not endpoints:
        st.error("No endpoints were discovered in the OpenAPI schema.")
        st.stop()

    st.sidebar.header("Available Endpoints")
    selection_label = st.sidebar.selectbox("Choose endpoint", [item["label"] for item in endpoints])
    selected = next(item for item in endpoints if item["label"] == selection_label)

    st.header(f"{selected['method']} {selected['path']}")
    operation = selected["operation"]
    if operation.get("description"):
        st.write(operation["description"])
    elif operation.get("summary"):
        st.write(operation["summary"])

    formatted_path = selected["path"]
    query_params: dict[str, Any] = {}
    header_params: dict[str, Any] = {}
    body_payload: dict[str, Any] = {}
    submitted = False

    if selected["method"] == "POST" and selected["path"] == "/screening/run":
        st.markdown("### Request Body")
        full_name = st.text_input("full_name")

        set_country = st.checkbox("Set country (optional)")
        country: str | None = None
        if set_country:
            country = st.text_input("country")

        set_dob = st.checkbox("Set date_of_birth (optional)")
        date_of_birth: str | None = None
        if set_dob:
            date_of_birth = st.text_input("date_of_birth")

        body_payload = {"full_name": full_name}
        if country:
            body_payload["country"] = country
        if date_of_birth:
            body_payload["date_of_birth"] = date_of_birth

        submitted = st.button("Send Request")
    else:
        with st.form(key=f"form_{selected['method']}_{selected['path']}"):
            formatted_path, query_params, header_params, _, body_payload = build_request_payload(
                selected, openapi_schema
            )
            submitted = st.form_submit_button("Send Request")

    if submitted:
        if selected["method"] == "POST" and selected["path"] == "/screening/run":
            full_name = body_payload.get("full_name", "")
            if not isinstance(full_name, str) or not full_name.strip():
                st.error("full_name is required.")
                st.stop()

        try:
            with st.spinner("Sending request..."):
                start_time = time_module.time()
                response = perform_request(
                    selected["method"],
                    formatted_path,
                    query_params,
                    header_params,
                    body_payload,
                )
                elapsed_seconds = time_module.time() - start_time
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")
            st.stop()

        if selected["method"] == "POST" and selected["path"] == "/screening/run":
            elapsed_minutes = round(elapsed_seconds / 60, 2)
            st.success(f"Search completed in {elapsed_minutes} minutes")

        st.write(f"Status Code: {response.status_code}")

        try:
            response_json = response.json()
        except ValueError:
            st.error("The API returned a non-JSON response.")
            st.code(response.text)
            st.stop()

        if response.ok:
            render_response_friendly(response_json)
        else:
            st.error("Request failed. See details below.")
            render_response_friendly(response_json)

        with st.expander("Developer View (Raw JSON)", expanded=False):
            st.json(response_json)


if __name__ == "__main__":
    main()
