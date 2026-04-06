"""
Entity context building for OASIS Agent Profile Generator

Assembles rich context from graph edges, related nodes, graph storage search,
and web enrichment — fed into LLM persona prompts.
"""

from typing import Dict, Any, Optional

from ..utils.logger import get_logger
from .entity_reader import EntityNode

logger = get_logger('miroshark.oasis_profile')


def build_entity_context(
    entity: EntityNode,
    storage,
    graph_id: Optional[str],
    web_enricher,
    simulation_requirement: str,
) -> str:
    """Build complete context string for an entity."""
    context_parts = []

    # 1. Entity attribute information
    if entity.attributes:
        attrs = [f"- {k}: {v}" for k, v in entity.attributes.items() if v and str(v).strip()]
        if attrs:
            context_parts.append("### Entity Attributes\n" + "\n".join(attrs))

    # 2. Related edge information (facts / relationships)
    existing_facts: set = set()
    if entity.related_edges:
        relationships = []
        for edge in entity.related_edges:
            fact = edge.get("fact", "")
            edge_name = edge.get("edge_name", "")
            direction = edge.get("direction", "")
            if fact:
                relationships.append(f"- {fact}")
                existing_facts.add(fact)
            elif edge_name:
                if direction == "outgoing":
                    relationships.append(f"- {entity.name} --[{edge_name}]--> (related entity)")
                else:
                    relationships.append(f"- (related entity) --[{edge_name}]--> {entity.name}")
        if relationships:
            context_parts.append("### Related Facts and Relationships\n" + "\n".join(relationships))

    # 3. Detailed information of related nodes
    if entity.related_nodes:
        related_info = []
        for node in entity.related_nodes:
            node_name = node.get("name", "")
            custom_labels = [l for l in node.get("labels", []) if l not in ["Entity", "Node"]]
            label_str = f" ({', '.join(custom_labels)})" if custom_labels else ""
            node_summary = node.get("summary", "")
            if node_summary:
                related_info.append(f"- **{node_name}**{label_str}: {node_summary}")
            else:
                related_info.append(f"- **{node_name}**{label_str}")
        if related_info:
            context_parts.append("### Related Entity Information\n" + "\n".join(related_info))

    # 4. Knowledge graph hybrid search
    graph_results = search_graph_for_entity(entity, storage, graph_id)
    if graph_results.get("facts"):
        new_facts = [f for f in graph_results["facts"] if f not in existing_facts]
        if new_facts:
            context_parts.append(
                "### Facts Retrieved from Knowledge Graph\n" +
                "\n".join(f"- {f}" for f in new_facts[:15])
            )
    if graph_results.get("node_summaries"):
        context_parts.append(
            "### Related Nodes Retrieved from Knowledge Graph\n" +
            "\n".join(f"- {s}" for s in graph_results["node_summaries"][:10])
        )

    # 5. Web enrichment
    existing_context = "\n\n".join(context_parts)
    entity_type = entity.get_entity_type() or "Entity"
    web_context = web_enricher.enrich_if_needed(
        entity_name=entity.name, entity_type=entity_type,
        existing_context=existing_context, simulation_requirement=simulation_requirement,
    )
    if web_context:
        context_parts.append(web_context)

    return "\n\n".join(context_parts)


def search_graph_for_entity(
    entity: EntityNode,
    storage,
    graph_id: Optional[str],
) -> Dict[str, Any]:
    """Use GraphStorage hybrid search (vector + BM25) to fetch rich entity context."""
    if not storage:
        return {"facts": [], "node_summaries": [], "context": ""}
    results: Dict[str, Any] = {"facts": [], "node_summaries": [], "context": ""}
    if not graph_id:
        return results

    entity_name = entity.name
    query = f"All information, activities, events, relationships and background about {entity_name}"
    try:
        edge_results = storage.search(graph_id=graph_id, query=query, limit=30, scope="edges")
        all_facts: set = set()
        if isinstance(edge_results, dict) and 'edges' in edge_results:
            for edge in edge_results['edges']:
                fact = edge.get('fact', '')
                if fact:
                    all_facts.add(fact)
        results["facts"] = list(all_facts)

        node_results = storage.search(graph_id=graph_id, query=query, limit=20, scope="nodes")
        all_summaries: set = set()
        if isinstance(node_results, dict) and 'nodes' in node_results:
            for node in node_results['nodes']:
                summary = node.get('summary', '')
                if summary:
                    all_summaries.add(summary)
                name = node.get('name', '')
                if name and name != entity_name:
                    all_summaries.add(f"Related Entity: {name}")
        results["node_summaries"] = list(all_summaries)

        context_parts = []
        if results["facts"]:
            context_parts.append("Fact Information:\n" + "\n".join(f"- {f}" for f in results["facts"][:20]))
        if results["node_summaries"]:
            context_parts.append("Related Entities:\n" + "\n".join(f"- {s}" for s in results["node_summaries"][:10]))
        results["context"] = "\n\n".join(context_parts)

        logger.info(
            f"Knowledge graph search: {entity_name}, "
            f"{len(results['facts'])} facts, {len(results['node_summaries'])} nodes"
        )
    except Exception as e:
        logger.warning(f"Knowledge graph search failed ({entity_name}): {e}")
    return results
