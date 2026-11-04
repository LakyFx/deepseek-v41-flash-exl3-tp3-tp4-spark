"""Disposable, offline JSON Schema validator; never imports the GPU engine."""
import json
import math
import sys



def strict_loads(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('duplicate_key')
            value[key] = item
        return value

    def constant(_):
        raise ValueError('non_finite_number')

    def floating(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError('non_finite_number')
        return value

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=floating)


def no_resource(uri):
    from referencing.exceptions import NoSuchResource
    # No network, filesystem, or implicit remote metaschema retrieval.
    raise NoSuchResource(ref=uri)


def nodes(schema):
    if not isinstance(schema, dict):
        return
    yield schema
    for key in ('properties', 'patternProperties', '$defs', 'definitions', 'dependentSchemas', 'dependencies'):
        if isinstance(schema.get(key), dict):
            for child in schema[key].values():
                yield from nodes(child)
    for key in ('allOf', 'anyOf', 'oneOf', 'prefixItems'):
        if isinstance(schema.get(key), list):
            for child in schema[key]:
                yield from nodes(child)
    for key in ('items', 'additionalItems', 'additionalProperties', 'unevaluatedProperties', 'unevaluatedItems',
                'contains', 'propertyNames', 'not', 'if', 'then', 'else'):
        child = schema.get(key)
        for sub in child if isinstance(child, list) else [child]:
            yield from nodes(sub)


def main():
    from jsonschema import FormatChecker, validators
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012
    try:
        body = strict_loads(sys.stdin.read(5 * 1024 * 1024 + 1))
        schema = body['schema']
        cls = validators.validator_for(schema)
        cls.check_schema(schema)
        registry = Registry(retrieve=no_resource)
        resolver = registry.resolver_with_root(Resource.from_contents(schema, default_specification=DRAFT202012))
        for node in nodes(schema):
            for key in ('$ref', '$dynamicRef', '$recursiveRef'):
                if key in node:
                    ref = node[key]
                    if not isinstance(ref, str) or not ref.startswith('#'):
                        raise ValueError('invalid_schema')
                    resolved = resolver.lookup(ref).contents
                    if not isinstance(resolved, (dict, bool)):
                        raise ValueError('invalid_schema')
        validator = cls(schema, registry=registry, format_checker=FormatChecker())
        for text in body.get('documents', []):
            value = strict_loads(text)
            if body.get('object_only') and not isinstance(value, dict):
                raise ValueError('root_not_object')
            # Do not sort/enumerate every error, nor return instance values.
            issue = next(validator.iter_errors(value), None)
            if issue is not None:
                print(json.dumps({'ok': False, 'reason': 'schema_mismatch', 'keyword': str(issue.validator)}))
                return
        print(json.dumps({'ok': True}))
    except Exception as exc:
        # Exception messages can contain client data. Only fixed categories leave.
        reason = 'invalid_schema' if 'documents' not in locals().get('body', {}) else 'invalid_json'
        if isinstance(exc, ValueError) and str(exc) in {'duplicate_key', 'non_finite_number', 'root_not_object'}:
            reason = str(exc)
        print(json.dumps({'ok': False, 'reason': reason}))


if __name__ == '__main__':
    main()
