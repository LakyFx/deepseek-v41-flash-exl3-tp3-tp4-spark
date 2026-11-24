"""Exercise the DeepSeek gateway with model responses at the native HTTP boundary."""
import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from laguna_gateway.app import create_app
from laguna_gateway.config import Settings
from laguna_gateway.engine import GatewayEngine, GatewayError
from laguna_gateway.metrics import GatewayMetrics


class CheckpointBackend:
    def __init__(self, count, invalid_attempts=0):
        self.count = count
        self.invalid_attempts = invalid_attempts
        self.operations = []
        self.output_limits = []
        self.system_prompts = []

    def __call__(self, request):
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 200})
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        self.output_limits.append(body["max_tokens"])
        system, source = [m["content"] for m in body["messages"]]
        self.system_prompts.append(system)
        data = json.loads(source)
        self.operations.append(data["operation"])
        schema = json.loads(system.splitlines()[-1])
        payload = {}
        for field, rule in schema["properties"].items():
            if "const" in rule:
                payload[field] = rule["const"]
            elif rule["type"] == "array":
                payload[field] = []
            else:
                payload[field] = {"text": "", "evidence_ids": []}

        citation = data["checkpoint_control"]["allowed_evidence_ids"][0]
        claims = [{"text": f"Claim {i}", "evidence_ids": [citation]} for i in range(self.count)]
        payload["facts"] = claims
        payload["hard_constraints"] = claims
        if self.invalid_attempts:
            self.invalid_attempts -= 1
            payload["facts"] = [{"text": "Unsupported", "evidence_ids": ["bad-id"]}]
        return httpx.Response(200, json={"choices": [{
            "message": {"role": "assistant", "content": json.dumps(payload)},
            "finish_reason": "stop",
        }]})


def settings():
    return Settings(
        served_model="deepseek-v4.1-flash",
        backend_family="qwen38_vllm",
        deepseek_v41_vllm=True,
        tensorfold_internal_json=True,
        chat_checkpoint_enforce=True,
        responses_checkpoint_mode="enforce",
        enforce_czech=False,
        compact_recent_tokens=64,
        responses_checkpoint_task_chars=4000,
    )


def history(count):
    # Each original is distinct. The model stub deliberately omits their wording.
    rules = [f"Pravidlo {i}. " + "Zachovej původní znění. " * 5 for i in range(count)]
    messages = [{"role": "user", "content": rule} for rule in rules]
    messages.extend([
        {"role": "assistant", "content": "live tail " * 300},
        {"role": "user", "content": "Pokračuj."},
    ])
    return messages, rules


async def compact(engine, route, messages):
    body = {"model": engine.settings.served_model, "max_tokens": 1024}
    if route == "chat":
        prepared = await engine.prepare({**body, "messages": messages}, force_compaction=True)
        text = prepared.body["messages"][0]["content"]
    else:
        body.pop("max_tokens")
        prepared = await engine.prepare_responses(
            {**body, "max_output_tokens": 1024, "input": messages}, force_compaction=True
        )
        text = prepared.body["input"][0]["content"][0]["text"]
    return json.loads(text.splitlines()[-1])


@pytest.mark.parametrize("route", ["chat", "responses"])
@pytest.mark.parametrize("count", [5, 12, 37])
def test_every_rule_reaches_the_next_request(route, count):
    async def run():
        messages, rules = history(count)
        backend = CheckpointBackend(count)
        async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(backend)) as client:
            engine = GatewayEngine(settings=settings(), client=client, vision_client=None, metrics=GatewayMetrics())
            try:
                state = await compact(engine, route, messages)
            finally:
                await engine.close_background_tasks()
        assert backend.operations == ["extract", "reconcile"]
        checkpoint = state["checkpoint"]
        assert len(checkpoint["facts"]) == len(checkpoint["hard_constraints"]) == count
        ledger = [json.loads(line) for line in state["critical_evidence_jsonl"].splitlines()]
        items = [record["item"] for record in ledger if "item" in record]
        assert [item["content"] for item in items if item["role"] == "user"] == rules
    asyncio.run(run())


@pytest.mark.parametrize("invalid_attempts", [1, 2])
def test_invalid_reference_is_retried_with_the_field_and_never_forwarded(invalid_attempts):
    async def run():
        messages, _ = history(5)
        backend = CheckpointBackend(5, invalid_attempts)
        async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(backend)) as client:
            engine = GatewayEngine(settings=settings(), client=client, vision_client=None, metrics=GatewayMetrics())
            try:
                if invalid_attempts == 1:
                    await compact(engine, "chat", messages)
                    assert backend.operations == ["extract", "extract", "reconcile"]
                else:
                    with pytest.raises(GatewayError) as exc:
                        await compact(engine, "chat", messages)
                    assert exc.value.code == "checkpoint_validation_failed"
                    assert backend.operations == ["extract", "extract"]
                assert "enum mismatch at $.facts[0].evidence_ids[0]" in backend.system_prompts[1]
            finally:
                await engine.close_background_tasks()
    asyncio.run(run())


@pytest.mark.parametrize("route", ["chat", "responses"])
@pytest.mark.parametrize("remaining", [2048, 0])
def test_internal_generation_respects_measured_token_room(route, remaining):
    async def run():
        cfg = settings()
        backend = CheckpointBackend(5)
        def upstream(request):
            if request.url.path == "/tokenize" and b"[INTERNAL_JSON_CONTRACT]" in request.content:
                prompt = cfg.chat_context_limit() - cfg.compact_safety_tokens - remaining
                return httpx.Response(200, json={"count": prompt})
            return backend(request)

        async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(upstream)) as client:
            engine = GatewayEngine(settings=cfg, client=client, vision_client=None, metrics=GatewayMetrics())
            messages, _ = history(5)
            try:
                if remaining:
                    await compact(engine, route, messages)
                    assert backend.output_limits == [remaining, remaining]
                else:
                    with pytest.raises(GatewayError) as exc:
                        await compact(engine, route, messages)
                    assert (exc.value.status_code, exc.value.code) == (400, "context_length_exceeded")
                    assert backend.operations == []
            finally:
                await engine.close_background_tasks()
    asyncio.run(run())


@pytest.mark.parametrize("route", ["chat/completions", "responses"])
def test_http_gateway_only_compacts_under_pressure_and_respects_the_context_limit(route):
    cfg = replace(settings(), api_key="local-test", upstream_base_url="http://test")
    backend = CheckpointBackend(13)
    delivered = []
    def upstream(request):
        data = json.loads(request.content)
        if request.url.path == "/tokenize":
            encoded = json.dumps(data)
            pressure = (
                "discardable-history" in encoded or "oversized-user-turn" in encoded
            ) and "[INTERNAL_JSON_CONTRACT]" not in encoded
            return httpx.Response(200, json={"count": cfg.max_model_len if pressure else 200})
        messages = data.get("messages") or []
        if messages and "[INTERNAL_JSON_CONTRACT]" in str(messages[0].get("content")):
            return backend(request)
        delivered.append(data)
        usage = {"prompt_tokens": 200, "completion_tokens": 1, "total_tokens": 201}
        if request.url.path == "/v1/responses":
            return httpx.Response(200, json={
                "id": "resp_test", "object": "response", "status": "completed",
                "output": [{"type": "message", "role": "assistant",
                            "content": [{"type": "output_text", "text": "OK"}]}],
                "usage": {"input_tokens": 200, "output_tokens": 1, "total_tokens": 201},
            })
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json={
            "id": "chatcmpl-test", "object": "chat.completion", "model": cfg.served_model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"},
                         "finish_reason": "stop"}], "usage": usage,
        })

    key = "messages" if route == "chat/completions" else "input"
    output_key = "max_tokens" if key == "messages" else "max_output_tokens"
    body = {"model": cfg.served_model, output_key: 1024}
    headers = {"Authorization": "Bearer local-test"}
    app = create_app(cfg, upstream_transport=httpx.MockTransport(upstream))
    with TestClient(app) as client:
        def submit(messages):
            return client.post(f"/v1/{route}", headers=headers, json={**body, key: messages})
        short = submit([{"role": "user", "content": "Hello"}])
        assert short.status_code == 200, short.text
        assert not backend.operations and len(delivered) == 1
        messages, rules = history(13)
        messages.insert(0, {"role": "assistant", "content": "discardable-history"})
        response = submit(messages)
        assert response.status_code == 200, response.text
        assert backend.operations == ["extract", "reconcile"] and len(delivered) == 2
        final = json.dumps(delivered[-1], ensure_ascii=False)
        assert all(rule in final for rule in rules)
        oversized = submit([{"role": "user", "content": "oversized-user-turn"}])
        assert oversized.status_code == 400, oversized.text
        assert oversized.json()["error"]["code"] == "context_length_exceeded"
        assert len(delivered) == 2
