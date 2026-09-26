from collections import defaultdict, deque

from app.EvoRAG.models import DependencyEdge, DependencyGraph, RetrievedEntity, StoredAttribute


SEMANTIC_ATTRIBUTE_TYPES = {"definition", "purpose", "core_idea", "mechanism", "components", "related"}
CONDITIONAL_ATTRIBUTE_TYPES = {"constraints"}


def build_dependency_graph(
    entities: list[RetrievedEntity],
    *,
    root_entity_id: int,
    max_edges_per_entity: int = 8,
) -> DependencyGraph:
    entity_by_id = {entity.id: entity for entity in entities}
    semantic_edges: list[DependencyEdge] = []
    conditional_edges: list[DependencyEdge] = []

    for entity in entities:
        edge_count = 0
        for attribute in entity.attributes:
            if edge_count >= max_edges_per_entity:
                break
            graph_type = graph_type_for_attribute(attribute)
            if graph_type is None:
                continue
            for target in mentioned_entities(attribute.value_text, entities, source_entity_id=entity.id):
                edge = DependencyEdge(
                    source_id=entity.id,
                    target_id=target.id,
                    relation=attribute.attr_type,
                    graph_type=graph_type,
                    evidence=attribute.value_text,
                    weight=max(0.1, attribute.confidence),
                )
                if graph_type == "conditional":
                    conditional_edges.append(edge)
                else:
                    semantic_edges.append(edge)
                edge_count += 1
                if edge_count >= max_edges_per_entity:
                    break

    if root_entity_id in entity_by_id and not semantic_edges and len(entities) > 1:
        for entity in entities:
            if entity.id == root_entity_id:
                continue
            semantic_edges.append(
                DependencyEdge(
                    source_id=root_entity_id,
                    target_id=entity.id,
                    relation="retrieved_related",
                    graph_type="semantic",
                    evidence="retrieved as a related entity",
                    weight=max(0.1, entity.score),
                )
            )

    all_edges = dedupe_edges([*semantic_edges, *conditional_edges])
    semantic_edges = [edge for edge in all_edges if edge.graph_type == "semantic"]
    conditional_edges = [edge for edge in all_edges if edge.graph_type == "conditional"]
    cycles = find_cycles(list(entity_by_id), all_edges)
    ordered_entity_ids = topological_order_with_root_first(list(entity_by_id), all_edges, root_entity_id)

    return DependencyGraph(
        root_entity_id=root_entity_id,
        entities=entities,
        semantic_edges=semantic_edges,
        conditional_edges=conditional_edges,
        ordered_entity_ids=ordered_entity_ids,
        cycles=cycles,
    )


def graph_type_for_attribute(attribute: StoredAttribute) -> str | None:
    if attribute.attr_type in CONDITIONAL_ATTRIBUTE_TYPES:
        return "conditional"
    if attribute.attr_type in SEMANTIC_ATTRIBUTE_TYPES:
        return "semantic"
    return None


def mentioned_entities(text: str, entities: list[RetrievedEntity], *, source_entity_id: int) -> list[RetrievedEntity]:
    normalized_text = normalize_match_text(text)
    if not normalized_text:
        return []

    matches: list[RetrievedEntity] = []
    for entity in entities:
        if entity.id == source_entity_id:
            continue
        names = [entity.canonical_name, *entity.aliases]
        for name in sorted(names, key=len, reverse=True):
            normalized_name = normalize_match_text(name)
            if len(normalized_name) < 2:
                continue
            if normalized_name in normalized_text:
                matches.append(entity)
                break
    return matches


def normalize_match_text(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def dedupe_edges(edges: list[DependencyEdge]) -> list[DependencyEdge]:
    deduped: dict[tuple[int, int, str, str], DependencyEdge] = {}
    for edge in edges:
        key = (edge.source_id, edge.target_id, edge.relation, edge.graph_type)
        existing = deduped.get(key)
        if existing is None or edge.weight > existing.weight:
            deduped[key] = edge
    return list(deduped.values())


def find_cycles(entity_ids: list[int], edges: list[DependencyEdge]) -> list[list[int]]:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.source_id].append(edge.target_id)

    index = 0
    stack: list[int] = []
    on_stack: set[int] = set()
    indexes: dict[int, int] = {}
    lowlinks: dict[int, int] = {}
    cycles: list[list[int]] = []

    def strongconnect(node_id: int) -> None:
        nonlocal index
        indexes[node_id] = index
        lowlinks[node_id] = index
        index += 1
        stack.append(node_id)
        on_stack.add(node_id)

        for target_id in adjacency.get(node_id, []):
            if target_id not in indexes:
                strongconnect(target_id)
                lowlinks[node_id] = min(lowlinks[node_id], lowlinks[target_id])
            elif target_id in on_stack:
                lowlinks[node_id] = min(lowlinks[node_id], indexes[target_id])

        if lowlinks[node_id] != indexes[node_id]:
            return

        component: list[int] = []
        while stack:
            target_id = stack.pop()
            on_stack.remove(target_id)
            component.append(target_id)
            if target_id == node_id:
                break
        if len(component) > 1:
            cycles.append(sorted(component))

    for entity_id in entity_ids:
        if entity_id not in indexes:
            strongconnect(entity_id)

    return cycles


def topological_order_with_root_first(entity_ids: list[int], edges: list[DependencyEdge], root_entity_id: int) -> list[int]:
    known_ids = set(entity_ids)
    indegree = {entity_id: 0 for entity_id in entity_ids}
    adjacency: dict[int, list[int]] = defaultdict(list)

    for edge in edges:
        if edge.source_id not in known_ids or edge.target_id not in known_ids:
            continue
        adjacency[edge.source_id].append(edge.target_id)
        indegree[edge.target_id] += 1

    ordered: list[int] = []
    seen: set[int] = set()
    queue: deque[int] = deque()
    if root_entity_id in known_ids:
        queue.append(root_entity_id)
    for entity_id in entity_ids:
        if indegree[entity_id] == 0 and entity_id != root_entity_id:
            queue.append(entity_id)

    while queue:
        entity_id = queue.popleft()
        if entity_id in seen:
            continue
        seen.add(entity_id)
        ordered.append(entity_id)
        for target_id in adjacency.get(entity_id, []):
            indegree[target_id] -= 1
            if indegree[target_id] <= 0:
                queue.append(target_id)

    for entity_id in entity_ids:
        if entity_id not in seen:
            ordered.append(entity_id)
    return ordered
