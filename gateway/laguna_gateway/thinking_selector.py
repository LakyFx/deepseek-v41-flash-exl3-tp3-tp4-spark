"""DeepSeek top-level selector compatibility; established kwargs retain priority."""
from typing import Any


def top_level_thinking(body: dict[str, Any], kwargs: dict[str, Any]) -> bool | None:
    if any(k in kwargs for k in ('thinking', 'enable_thinking', 'thinking_mode')):
        return None
    if 'thinking' not in body:
        return None
    value = body['thinking']
    if isinstance(value, bool):
        return value
    if isinstance(value, dict) and value.get('type') in ('enabled', 'disabled'):
        return value['type'] == 'enabled'
    raise ValueError('thinking must be a bool or {"type":"enabled"|"disabled"}')
