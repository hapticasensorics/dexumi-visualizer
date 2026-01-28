from __future__ import annotations

SCHEMA_VERSION = 1

PLUGIN_MANIFEST_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "DexUMI Plugin Manifest",
    "type": "object",
    "required": ["schema_version", "id", "name", "version", "entrypoint"],
    "properties": {
        "schema_version": {"type": "integer", "minimum": 1},
        "id": {"type": "string", "minLength": 1},
        "name": {"type": "string", "minLength": 1},
        "version": {"type": "string", "minLength": 1},
        "description": {"type": "string"},
        "entrypoint": {
            "type": "object",
            "required": ["module", "callable"],
            "properties": {
                "module": {"type": "string", "minLength": 1},
                "callable": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
        "formats": {"type": "array", "items": {"type": "string"}},
        "capabilities": {"type": "array", "items": {"type": "string"}},
        "authors": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
                "additionalProperties": True,
            },
        },
        "homepage": {"type": "string"},
        "license": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "metadata": {"type": "object"},
    },
    "additionalProperties": False,
}
