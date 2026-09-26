import hashlib
import re
from collections import OrderedDict

from app.EvoRAG.models import BlockEntityExtraction, ExtractedEntity
from app.EvoRAG.entity_store.models import EntityAttributeInput, EntityScope, IncomingEntity


def normalize_name(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[\s\-_./()（）【】\[\]{}:：,，;；]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_fingerprint(value: str) -> str:
    normalized = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def entity_fingerprint(entity: ExtractedEntity) -> str:
    return f"{normalize_name(entity.name)}::{normalize_name(entity.entity_type)}"


def dedupe_extracted_entities(
    blocks: list[BlockEntityExtraction],
    *,
    scope: EntityScope | None = None,
) -> list[IncomingEntity]:
    grouped: OrderedDict[str, IncomingEntity] = OrderedDict()
    entity_scope = scope or EntityScope()

    for block_result in blocks:
        block = block_result.block
        for entity in block_result.entities:
            key = entity_fingerprint(entity)
            incoming = grouped.get(key)
            if incoming is None:
                incoming = IncomingEntity(
                    name=entity.name,
                    normalized_name=normalize_name(entity.name),
                    entity_type=entity.entity_type or "concept",
                    scope=entity_scope,
                    aliases=list(entity.aliases),
                    identity_description=entity.identity_description,
                    attributes=[],
                    source_count=0,
                )
                grouped[key] = incoming
            else:
                incoming.aliases = merge_unique([*incoming.aliases, *entity.aliases])
                if not incoming.identity_description and entity.identity_description:
                    incoming.identity_description = entity.identity_description

            incoming.source_count += 1
            incoming.attributes = merge_attributes(
                incoming.attributes,
                entity_to_attributes(entity, block_index=block.block_index),
            )

    for incoming in grouped.values():
        incoming.description_for_match = build_description_for_match(incoming)

    return list(grouped.values())


def entity_to_attributes(entity: ExtractedEntity, *, block_index: int) -> list[EntityAttributeInput]:
    attributes: list[EntityAttributeInput] = []
    for attr_type, value in entity.attributes.iter_values():
        if not value.value:
            continue
        attributes.append(
            EntityAttributeInput(
                attr_type=attr_type,
                value_text=value.value,
                evidence=value.evidence,
                confidence=value.confidence,
                block_index=block_index,
            )
        )
    return attributes


def merge_attributes(
    existing: list[EntityAttributeInput],
    incoming: list[EntityAttributeInput],
) -> list[EntityAttributeInput]:
    merged: OrderedDict[str, EntityAttributeInput] = OrderedDict()
    for item in [*existing, *incoming]:
        key = f"{item.attr_type}:{text_fingerprint(item.value_text)}"
        if key not in merged:
            merged[key] = item
        elif item.evidence and item.evidence not in merged[key].evidence:
            merged[key].evidence = " | ".join(part for part in [merged[key].evidence, item.evidence] if part)
    return list(merged.values())


def merge_unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = str(value or "").strip()
        key = normalize_name(clean)
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return result


def build_description_for_match(entity: IncomingEntity) -> str:
    parts = [
        f"name: {entity.name}",
        f"type: {entity.entity_type}",
    ]
    if entity.aliases:
        parts.append(f"aliases: {', '.join(entity.aliases[:8])}")
    if entity.identity_description:
        parts.append(f"identity: {entity.identity_description}")
    for attr_type in ("definition", "purpose", "core_idea", "mechanism", "components", "constraints", "related"):
        values = [item.value_text for item in entity.attributes if item.attr_type == attr_type]
        if values:
            parts.append(f"{attr_type}: {'; '.join(values[:5])}")
    return "\n".join(parts)
