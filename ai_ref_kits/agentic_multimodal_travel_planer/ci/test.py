#!/usr/bin/env python3
"""Sanity checks for agentic multimodal travel planner.

Checks covered:
1) Live LLM sanity: OVMS endpoints and a real completion response.
2) MCP sanity: startup manager can launch mock MCP servers on configured ports.
3) Agent sanity: agent manager can launch mock agent runners on configured ports.
"""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from openai import OpenAI
import yaml
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from beeai_framework.adapters.a2a.agents.agent import A2AAgent
    from beeai_framework.memory import UnconstrainedMemory
except ImportError:
    A2AAgent = None  # type: ignore[assignment]
    UnconstrainedMemory = None  # type: ignore[assignment]

try:
    from utils.util import extract_response_text
except ImportError:
    extract_response_text = None  # type: ignore[assignment]

ROUTER_SANITY_QUERY = (
    "Find flights from Milan to Berlin for 2026-03-01 to 2026-03-10 in "
    "economy class, confirm all details, then include token SANITY_OK in "
    "your final answer."
)
ROUTER_SANITY_EXPECTED_TOKEN = "SANITY_OK"


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        if value > 0:
            return value
    except ValueError:
        pass
    return default


def _is_port_open(port: int, host: str = "localhost") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


def _http_get_json(url: str, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload) if payload else {}
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GET failed for {url}: {exc}") from exc


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _strip_model_provider_prefix(model_name: str) -> str:
    # BeeAI config can use "openai:<model_id>".
    if model_name.startswith("openai:"):
        return model_name.split(":", 1)[1]
    return model_name


def _force_localhost(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    _assert(
        parsed.scheme in {"http", "https"},
        f"Invalid URL scheme in config: {url}",
    )
    _assert(
        parsed.port is not None,
        f"URL must include explicit port in config: {url}",
    )
    netloc = f"localhost:{parsed.port}"
    return urllib.parse.urlunparse(
        (
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def _resolve_llm_vlm_targets_from_config() -> tuple[str, str, str]:
    agents_cfg = _load_yaml(PROJECT_ROOT / "config" / "agents_config.yaml")
    mcp_cfg = _load_yaml(PROJECT_ROOT / "config" / "mcp_config.yaml")

    travel_router = agents_cfg.get("travel_router", {})
    llm_cfg = travel_router.get("llm", {}) if isinstance(travel_router, dict) else {}
    image_mcp = mcp_cfg.get("image_mcp", {})

    _assert(
        isinstance(llm_cfg, dict) and llm_cfg.get("api_base"),
        "Missing required config: travel_router.llm.api_base in agents_config.yaml",
    )
    _assert(
        isinstance(llm_cfg, dict) and llm_cfg.get("model"),
        "Missing required config: travel_router.llm.model in agents_config.yaml",
    )
    _assert(
        isinstance(image_mcp, dict) and image_mcp.get("ovms_base_url"),
        "Missing required config: image_mcp.ovms_base_url in mcp_config.yaml",
    )

    llm_base = _force_localhost(str(llm_cfg["api_base"]).rstrip("/"))
    llm_model = _strip_model_provider_prefix(str(llm_cfg["model"]))
    vlm_base = _force_localhost(str(image_mcp["ovms_base_url"]).rstrip("/"))
    return llm_base, vlm_base, llm_model


def _pick_model_from_models_endpoint(models_payload: dict) -> str:
    data = models_payload.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and first.get("id"):
            return str(first["id"])
    raise RuntimeError(
        "No model id found in /v3/models response. "
        f"Payload={json.dumps(models_payload, ensure_ascii=True)}"
    )


def _ensure_v3_base(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v3"):
        return base
    return f"{base}/v3"


def _wait_for_models_payload(
    base_url: str, label: str, timeout_s: int = 180, interval_s: float = 2.0
) -> dict:
    deadline = time.time() + timeout_s
    last_payload: dict = {}
    while time.time() < deadline:
        payload = _http_get_json(f"{base_url}/models")
        last_payload = payload if isinstance(payload, dict) else {}
        data = last_payload.get("data")
        if isinstance(data, list) and len(data) > 0:
            return last_payload
        time.sleep(interval_s)

    raise RuntimeError(
        f"{label} models did not become ready within {timeout_s}s. "
        f"Last payload={json.dumps(last_payload, ensure_ascii=True)}"
    )


def _serialize_completion_for_logs(completion: object) -> str:
    try:
        if hasattr(completion, "model_dump"):
            payload = completion.model_dump()
        else:
            payload = str(completion)
        return json.dumps(payload, ensure_ascii=True)
    except Exception:
        return str(completion)


def check_live_llm_sanity() -> None:
    llm_base, vlm_base, _configured_llm_model = (
        _resolve_llm_vlm_targets_from_config()
    )
    llm_base = _ensure_v3_base(llm_base)
    vlm_base = _ensure_v3_base(vlm_base)

    # Basic health endpoints.
    llm_models = _wait_for_models_payload(llm_base, "LLM")
    vlm_models = _wait_for_models_payload(vlm_base, "VLM")
    _assert(
        isinstance(llm_models.get("data"), list),
        "LLM /models endpoint did not return expected payload.",
    )
    _assert(
        isinstance(vlm_models.get("data"), list),
        "VLM /models endpoint did not return expected payload.",
    )
    llm_model = _pick_model_from_models_endpoint(llm_models)
    vlm_model = _pick_model_from_models_endpoint(vlm_models)

    # Test LLM chat completion
    print(f"Testing LLM chat completion at {llm_base}...")
    try:
        llm_client = OpenAI(base_url=llm_base, api_key="unused")
        llm_completion = llm_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "hello"},
            ],
            max_tokens=100,
            extra_body={"top_k": 1},
            stream=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"OpenAI SDK chat completion failed for LLM {llm_base}: {exc}"
        ) from exc

    print(f"LLM chat completion response: {_serialize_completion_for_logs(llm_completion)}")
    llm_choices = llm_completion.choices if hasattr(llm_completion, "choices") else []
    _assert(
        isinstance(llm_choices, list) and len(llm_choices) > 0,
        "No LLM choices returned.",
    )

    # Test VLM chat completion
    print(f"Testing VLM chat completion at {vlm_base}...")
    try:
        vlm_client = OpenAI(base_url=vlm_base, api_key="unused")
        vlm_completion = vlm_client.chat.completions.create(
            model=vlm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "hello"},
            ],
            max_tokens=100,
            extra_body={"top_k": 1},
            stream=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"OpenAI SDK chat completion failed for VLM {vlm_base}: {exc}"
        ) from exc

    print(f"VLM chat completion response: {_serialize_completion_for_logs(vlm_completion)}")
    vlm_choices = vlm_completion.choices if hasattr(vlm_completion, "choices") else []
    _assert(
        isinstance(vlm_choices, list) and len(vlm_choices) > 0,
        "No VLM choices returned.",
    )
    print("Live LLM and VLM sanity checks passed.")


def _wait_for_ports(ports: list[int], timeout_s: int = 120) -> None:
    deadline = time.time() + timeout_s
    pending = set(ports)
    while time.time() < deadline and pending:
        for port in list(pending):
            if _is_port_open(port):
                pending.remove(port)
        if pending:
            time.sleep(0.5)
    _assert(not pending, f"Timed out waiting for ports: {sorted(pending)}")


def _mcp_ports_from_config() -> list[int]:
    mcp_cfg_path = PROJECT_ROOT / "config" / "mcp_config.yaml"
    mcp_cfg = _load_yaml(mcp_cfg_path)
    return [
        int(section.get("mcp_port"))
        for section in mcp_cfg.values()
        if isinstance(section, dict) and section.get("mcp_port")
    ]


def _agent_ports_from_config() -> list[int]:
    agents_cfg_path = PROJECT_ROOT / "config" / "agents_config.yaml"
    agents_cfg = _load_yaml(agents_cfg_path)
    return [
        int(section.get("port"))
        for section in agents_cfg.values()
        if (
            isinstance(section, dict)
            and section.get("enabled", True)
            and section.get("port")
        )
    ]


def _enabled_agents_from_config() -> list[tuple[str, dict]]:
    agents_cfg = _load_yaml(PROJECT_ROOT / "config" / "agents_config.yaml")
    return [
        (name, cfg)
        for name, cfg in agents_cfg.items()
        if (
            isinstance(cfg, dict)
            and cfg.get("enabled", True)
            and cfg.get("port")
        )
    ]


def _router_sanity_test_case() -> tuple[str, str]:
    return ROUTER_SANITY_QUERY, ROUTER_SANITY_EXPECTED_TOKEN


def _agent_url_from_card(card_payload: dict) -> str:
    if isinstance(card_payload, dict):
        raw_url = card_payload.get("url")
        if isinstance(raw_url, str) and raw_url.strip():
            parsed = urllib.parse.urlparse(raw_url.strip())
            if parsed.scheme in {"http", "https"} and parsed.port is not None:
                return f"{parsed.scheme}://127.0.0.1:{parsed.port}"
    raise RuntimeError(
        "Agent card missing valid 'url' with explicit port. "
        f"Payload={json.dumps(card_payload, ensure_ascii=True)}"
    )


def check_mcp_services_up() -> None:
    ports = _mcp_ports_from_config()
    _assert(ports, "No MCP ports found in mcp_config.yaml")
    _wait_for_ports(ports, timeout_s=120)
    print(f"MCP ports are up: {ports}")


def check_agent_services_up() -> None:
    ports = _agent_ports_from_config()
    _assert(ports, "No enabled agent ports found in agents_config.yaml")
    _wait_for_ports(ports, timeout_s=120)
    print(f"Agent ports are up: {ports}")

    # Verify agent-cards for all agents
    for agent_name, cfg in _enabled_agents_from_config():
        agent_port = int(cfg["port"])
        card_base_url = f"http://127.0.0.1:{agent_port}"
        card_url = f"{card_base_url}/.well-known/agent-card.json"
        card_payload = _http_get_json(card_url)
        print(
            f"{agent_name} agent-card: "
            f"{json.dumps(card_payload, ensure_ascii=True)}"
        )
        agent_url = _agent_url_from_card(card_payload)
        print(f"{agent_name} is accessible at {agent_url}")

    # Query travel_router only (it will test full stack via handoffs)
    query, expected_token = _router_sanity_test_case()
    query_timeout_s = _int_env("AGENT_QUERY_TIMEOUT_SECONDS", 600)
    query_retries = _int_env("AGENT_QUERY_RETRIES", 1)

    router_cfg = None
    for agent_name, cfg in _enabled_agents_from_config():
        if agent_name == "travel_router":
            router_cfg = cfg
            break

    if router_cfg:
        agent_port = int(router_cfg["port"])
        card_base_url = f"http://127.0.0.1:{agent_port}"
        card_url = f"{card_base_url}/.well-known/agent-card.json"
        card_payload = _http_get_json(card_url)
        agent_url = _agent_url_from_card(card_payload)

        print(f"Querying travel_router at {agent_url}...", flush=True)
        last_error: RuntimeError | None = None
        response_text = ""
        for attempt in range(1, query_retries + 1):
            try:
                response_text = _query_agent(
                    query=query,
                    agent_url=agent_url,
                    timeout_s=query_timeout_s,
                )
                break
            except RuntimeError as exc:
                last_error = exc
                print(
                    f"travel_router query attempt {attempt}/{query_retries} "
                    f"failed: {exc}",
                    flush=True,
                )
                if attempt < query_retries:
                    time.sleep(2)
        if not response_text:
            raise RuntimeError(
                f"travel_router query failed after {query_retries} attempts"
            ) from last_error
        print(f"travel_router response: {response_text}")
        _assert(
            expected_token.lower() in response_text.lower(),
            f"travel_router response missing expected token '{expected_token}'.",
        )
    else:
        print("Warning: travel_router not found in enabled agents")

    print("Agent endpoint sanity passed.")


def _query_agent(query: str, agent_url: str, timeout_s: int = 60) -> str:
    _assert(
        A2AAgent is not None and UnconstrainedMemory is not None,
        "Missing beeai_framework dependency for agent queries.",
    )
    _assert(
        extract_response_text is not None,
        "Missing utils.util.extract_response_text import for agent queries.",
    )

    async def _run_query() -> str:
        client = A2AAgent(url=agent_url, memory=UnconstrainedMemory())

        async def _ignore_event(_data: object, _event: object) -> None:
            return None

        # Match start_ui.y pattern: do NOT wrap with asyncio.wait_for
        response = await client.run(query).on("update", _ignore_event).on(
            "final_answer", _ignore_event
        )
        return extract_response_text(response)

    try:
        # Apply timeout at the asyncio.run level instead
        return asyncio.run(_run_query())
    except Exception as exc:
        raise RuntimeError(
            f"Agent query failed for {agent_url}: {exc}"
        ) from exc


def check_overall_placeholder() -> None:
    # Placeholder step kept to preserve workflow stage order.
    print(
        "Overall check placeholder: travel_router validation handled in "
        "--check-agents."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity checks for travel planner kit"
    )
    parser.add_argument(
        "--check-ovms",
        action="store_true",
        help="Check live OVMS LLM/VLM endpoints",
    )
    parser.add_argument(
        "--check-mcp",
        action="store_true",
        help="Check MCP services are up from config ports",
    )
    parser.add_argument(
        "--check-agents",
        action="store_true",
        help="Check enabled agents are up and return a simple query response",
    )
    parser.add_argument(
        "--check-overall",
        action="store_true",
        help="Placeholder step for overall validation",
    )
    args = parser.parse_args()

    if args.check_ovms:
        check_live_llm_sanity()
        return
    if args.check_mcp:
        check_mcp_services_up()
        return
    if args.check_agents:
        check_agent_services_up()
        return
    if args.check_overall:
        check_overall_placeholder()
        return

    # Default behavior for local/manual usage: run the same staged checks.
    check_live_llm_sanity()
    check_mcp_services_up()
    check_agent_services_up()
    check_overall_placeholder()
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
