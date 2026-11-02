"""Tool-safe final delivery for buffered image-bearing agent turns.

No tools are executed here. Partial generations are never exposed, so a bounded
repair cannot cause a client to execute a tool from the discarded attempt.
"""
from copy import deepcopy
import json


def tool_names(body):
    if body.get('tool_choice') == 'none':
        return set()
    names = set()
    for tool in body.get('tools') or []:
        if not isinstance(tool, dict): continue
        namespace = tool.get('name') if tool.get('type') == 'namespace' else None
        for child in tool.get('tools',[]) if namespace else [tool]:
            definition = child.get('function',child) if child.get('type') == 'function' else child
            if isinstance(definition, dict) and isinstance(definition.get('name'), str):
                names.add((namespace,definition['name']))
    choice = body.get('tool_choice')
    if isinstance(choice, dict):
        selected = choice.get('function', choice)
        if isinstance(selected, dict) and selected.get('name'):
            names.intersection_update({(selected.get('namespace'),selected['name'])})
    return names


def valid_call(identity, name, arguments, names, namespace=None):
    if not isinstance(identity, str) or not identity or (namespace,name) not in names:
        return False
    if not isinstance(arguments, str): return False
    try:
        return isinstance(json.loads(arguments), dict)
    except (ValueError, TypeError):
        return False


def response_events(data):
    """Replay one already-validated Responses result with stable item/call IDs."""
    sequence = 0

    def frame(kind, **fields):
        nonlocal sequence
        event = dict(type=kind, sequence_number=sequence, **fields)
        sequence += 1
        return ('data: ' + json.dumps(event, ensure_ascii=False, separators=(',', ':')) + '\n\n').encode()

    shell = {**data, 'status':'in_progress', 'output':[]}
    yield frame('response.created', response=shell)
    yield frame('response.in_progress', response=shell)
    for index, item in enumerate(data.get('output') or []):
        added = deepcopy(item)
        added['status'] = 'in_progress'
        if item['type'] == 'message': added['content'] = []
        if item['type'] == 'function_call': added['arguments'] = ''
        if item['type'] == 'custom_tool_call': added['input'] = ''
        yield frame('response.output_item.added', output_index=index, item=added)
        common = dict(item_id=item['id'], output_index=index)
        if item['type'] == 'message':
            for content_index, part in enumerate(item.get('content') or []):
                part_fields = dict(**common, content_index=content_index)
                start = {**part}
                field = 'refusal' if part.get('type') == 'refusal' else 'text'
                start[field] = ''
                yield frame('response.content_part.added', **part_fields, part=start)
                if field in part:
                    kind = 'response.refusal' if field == 'refusal' else 'response.output_text'
                    yield frame(kind+'.delta', **part_fields, delta=part[field])
                    yield frame(kind+'.done', **part_fields, **{field:part[field]})
                yield frame('response.content_part.done', **part_fields, part=part)
        elif item['type'] in {'function_call','custom_tool_call'}:
            field = 'arguments' if item['type'] == 'function_call' else 'input'
            kind = 'response.function_call_arguments' if field == 'arguments' else 'response.custom_tool_call_input'
            yield frame(kind+'.delta', **common, delta=item[field])
            yield frame(kind+'.done', **common, **{field:item[field]})
        yield frame('response.output_item.done', output_index=index, item=item)
    yield frame('response.completed', response=data)
