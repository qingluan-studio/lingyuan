#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 灵元模型项目 (LingYuan Model Project) — 第 25 模块
 知识图谱增强系统 (Knowledge Graph Augmented System)
================================================================================

模块概述:
    本模块实现了知识图谱增强的检索增强生成(RAG)系统。
    通过知识图谱提供结构化知识, 结合向量检索实现精准的知识注入。

核心组件:
    1. Entity             — 实体
    2. Relation           — 关系
    3. KnowledgeGraph     — 知识图谱 (实体/关系/三元组管理)
    4. GraphReasoner      — 图推理器 (路径搜索/多跳推理)
    5. VectorIndex        — 向量索引 (HNSW简化版)
    6. KnowledgeRetriever — 知识检索器 (向量+图谱混合检索)
    7. ContextBuilder     — 上下文构建器 (组装检索结果)
    8. RAGEngine          — RAG引擎 (检索增强生成)
    9. EntityLinker       — 实体链接器 (文本->图谱实体)
   10. FactChecker        — 事实检查器 (基于图谱验证)
   11. KnowledgeUpdater   — 知识更新器 (增量更新图谱)
   12. GraphEmbedder      — 图谱嵌入器 (TransE)
   13. SubgraphExtractor  — 子图提取器
   14. MultiHopReasoner   — 多跳推理器
   15. KnowledgePipeline  — 知识流水线 (端到端)

作者: 灵元模型项目组
版本: 1.0.0
================================================================================
"""

import os
import sys
import math
import time
import json
import random
import hashlib
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple, Set


# ============================================================
# 枚举
# ============================================================

class EntityType(Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    TIME = "time"
    OBJECT = "object"
    UNKNOWN = "unknown"


class RelationType(Enum):
    IS_A = "is_a"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    CAUSED_BY = "caused_by"
    LOCATED_IN = "located_in"
    CREATED_BY = "created_by"
    MEMBER_OF = "member_of"
    HAS_PROPERTY = "has_property"
    SIMILAR_TO = "similar_to"
    OCCURRED_AT = "occurred_at"


class RetrievalStrategy(Enum):
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


# ============================================================
# 实体与关系
# ============================================================

@dataclass
class Entity:
    """知识图谱实体"""
    eid: str
    name: str
    entity_type: EntityType = EntityType.UNKNOWN
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    aliases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "eid": self.eid, "name": self.name,
            "type": self.entity_type.value, "properties": self.properties,
            "aliases": self.aliases,
        }


@dataclass
class Relation:
    """知识图谱关系 (三元组)"""
    rid: str
    head: str  # 实体ID
    relation: str
    tail: str  # 实体ID
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "rid": self.rid, "head": self.head,
            "relation": self.relation, "tail": self.tail,
            "confidence": self.confidence,
        }


@dataclass
class KnowledgeChunk:
    """知识文本块"""
    cid: str
    text: str
    source: str = ""
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    linked_entities: List[str] = field(default_factory=list)


# ============================================================
# 知识图谱
# ============================================================

class KnowledgeGraph:
    """知识图谱 — 管理实体和关系

    支持:
    - 实体/关系的增删查改
    - 三元组查询
    - 邻居查找
    - 子图提取
    """

    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self._relation_index: Dict[str, List[Relation]] = defaultdict(list)  # head -> relations
        self._tail_index: Dict[str, List[Relation]] = defaultdict(list)  # tail -> relations
        self._name_index: Dict[str, str] = {}  # name -> eid
        self._alias_index: Dict[str, str] = {}  # alias -> eid

    def add_entity(self, entity: Entity) -> bool:
        if entity.eid in self.entities:
            return False
        self.entities[entity.eid] = entity
        self._name_index[entity.name] = entity.eid
        for alias in entity.aliases:
            self._alias_index[alias] = entity.eid
        return True

    def add_relation(self, relation: Relation) -> bool:
        if relation.head not in self.entities or relation.tail not in self.entities:
            return False
        self.relations.append(relation)
        self._relation_index[relation.head].append(relation)
        self._tail_index[relation.tail].append(relation)
        return True

    def get_entity(self, eid: str) -> Optional[Entity]:
        return self.entities.get(eid)

    def find_entity_by_name(self, name: str) -> Optional[Entity]:
        eid = self._name_index.get(name) or self._alias_index.get(name)
        if eid:
            return self.entities.get(eid)
        return None

    def get_neighbors(self, eid: str, direction: str = "both") -> List[str]:
        """获取邻居实体"""
        neighbors = set()
        if direction in ("both", "out"):
            for rel in self._relation_index.get(eid, []):
                neighbors.add(rel.tail)
        if direction in ("both", "in"):
            for rel in self._tail_index.get(eid, []):
                neighbors.add(rel.head)
        return list(neighbors)

    def get_relations_of(self, eid: str) -> List[Relation]:
        """获取实体的所有关系"""
        return self._relation_index.get(eid, []) + self._tail_index.get(eid, [])

    def query_triples(self, head: Optional[str] = None,
                      relation: Optional[str] = None,
                      tail: Optional[str] = None) -> List[Relation]:
        """查询三元组"""
        results = []
        for rel in self.relations:
            if head and rel.head != head:
                continue
            if relation and rel.relation != relation:
                continue
            if tail and rel.tail != tail:
                continue
            results.append(rel)
        return results

    def extract_subgraph(self, eid: str, depth: int = 2) -> Dict:
        """提取以eid为中心的子图"""
        visited = set()
        entities = []
        relations = []

        queue = deque([(eid, 0)])
        while queue:
            curr, d = queue.popleft()
            if curr in visited or d > depth:
                continue
            visited.add(curr)

            if curr in self.entities:
                entities.append(self.entities[curr].to_dict())

            if d < depth:
                for rel in self._relation_index.get(curr, []):
                    relations.append(rel.to_dict())
                    if rel.tail not in visited:
                        queue.append((rel.tail, d + 1))
                for rel in self._tail_index.get(curr, []):
                    relations.append(rel.to_dict())
                    if rel.head not in visited:
                        queue.append((rel.head, d + 1))

        return {
            "center": eid,
            "entities": entities,
            "relations": relations,
            "depth": depth,
        }

    def get_stats(self) -> Dict:
        return {
            "num_entities": len(self.entities),
            "num_relations": len(self.relations),
            "entity_types": dict(defaultdict(int, {
                t.value: sum(1 for e in self.entities.values() if e.entity_type == t)
                for t in EntityType
            })),
        }


# ============================================================
# 图推理器
# ============================================================

class GraphReasoner:
    """图推理器 — 基于图谱的推理

    支持路径搜索和多跳推理
    """

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    def find_paths(self, start: str, end: str, max_depth: int = 4) -> List[List[str]]:
        """查找从start到end的所有路径 (BFS)"""
        if start not in self.graph.entities or end not in self.graph.entities:
            return []

        paths = []
        queue = deque([(start, [start])])
        visited_paths = set()

        while queue:
            curr, path = queue.popleft()
            if len(path) > max_depth:
                continue
            if curr == end and len(path) > 1:
                paths.append(path)
                continue
            if curr in visited_paths:
                continue
            visited_paths.add(curr)

            neighbors = self.graph.get_neighbors(curr)
            for n in neighbors:
                if n not in path:
                    queue.append((n, path + [n]))

        return paths[:10]  # 限制结果数

    def multi_hop_query(self, start: str, hops: int = 2) -> Dict:
        """多跳查询: 从start出发, hops跳内的所有可达实体"""
        if start not in self.graph.entities:
            return {"start": start, "results": []}

        visited = {start: 0}
        queue = deque([(start, 0)])

        while queue:
            curr, d = queue.popleft()
            if d >= hops:
                continue
            for n in self.graph.get_neighbors(curr):
                if n not in visited:
                    visited[n] = d + 1
                    queue.append((n, d + 1))

        results = []
        for eid, dist in visited.items():
            if eid != start:
                entity = self.graph.get_entity(eid)
                results.append({
                    "entity": entity.name if entity else eid,
                    "eid": eid,
                    "distance": dist,
                })

        results.sort(key=lambda x: x["distance"])
        return {"start": start, "results": results}

    def infer_relation(self, head: str, tail: str) -> List[Dict]:
        """推断两个实体间的关系"""
        paths = self.find_paths(head, tail, max_depth=3)
        inferences = []

        for path in paths:
            relations_chain = []
            for i in range(len(path) - 1):
                rels = self.graph.query_triples(head=path[i], tail=path[i + 1])
                if rels:
                    relations_chain.append(rels[0].relation)
                else:
                    rels = self.graph.query_triples(head=path[i + 1], tail=path[i])
                    if rels:
                        relations_chain.append(f"reverse_{rels[0].relation}")
                    else:
                        relations_chain.append("unknown")

            inferences.append({
                "path": path,
                "relations": relations_chain,
                "length": len(path) - 1,
            })

        return inferences


# ============================================================
# 向量索引
# ============================================================

class VectorIndex:
    """向量索引 — 简化版HNSW

    支持向量插入和近似最近邻搜索
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.vectors: List[List[float]] = []
        self.ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}

    def add(self, item_id: str, vector: List[float]) -> None:
        if item_id in self._id_to_idx:
            self.vectors[self._id_to_idx[item_id]] = vector
        else:
            self._id_to_idx[item_id] = len(self.vectors)
            self.vectors.append(vector)
            self.ids.append(item_id)

    def search(self, query: List[float], top_k: int = 5) -> List[Dict]:
        """暴力搜索最近邻"""
        if not self.vectors:
            return []

        scores = []
        for i, vec in enumerate(self.vectors):
            sim = self._cosine_sim(query, vec)
            scores.append((sim, i))

        scores.sort(key=lambda x: -x[0])
        return [{"id": self.ids[idx], "score": round(score, 4)}
                for score, idx in scores[:top_k]]

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na > 0 and nb > 0 else 0.0

    def size(self) -> int:
        return len(self.vectors)


# ============================================================
# 知识检索器
# ============================================================

class KnowledgeRetriever:
    """知识检索器 — 向量+图谱混合检索

    策略:
    - VECTOR: 纯向量检索
    - GRAPH: 纯图谱检索
    - HYBRID: 混合检索 (向量找种子, 图谱扩展)
    """

    def __init__(self, graph: KnowledgeGraph, vector_index: VectorIndex,
                 embed_fn: Optional[Callable] = None):
        self.graph = graph
        self.vector_index = vector_index
        self.embed_fn = embed_fn or self._default_embed
        self.reasoner = GraphReasoner(graph)

    def retrieve(self, query: str, top_k: int = 5,
                 strategy: RetrievalStrategy = RetrievalStrategy.HYBRID) -> Dict:
        """检索知识"""
        query_vec = self.embed_fn(query)

        if strategy == RetrievalStrategy.VECTOR:
            return self._vector_retrieve(query_vec, top_k)
        elif strategy == RetrievalStrategy.GRAPH:
            return self._graph_retrieve(query, top_k)
        else:
            return self._hybrid_retrieve(query, query_vec, top_k)

    def _vector_retrieve(self, query_vec: List[float], top_k: int) -> Dict:
        results = self.vector_index.search(query_vec, top_k)
        return {
            "strategy": "vector",
            "results": results,
            "total": len(results),
        }

    def _graph_retrieve(self, query: str, top_k: int) -> Dict:
        # 查找查询中提到的实体
        entity = self.graph.find_entity_by_name(query)
        if not entity:
            # 模糊匹配
            candidates = []
            for e in self.graph.entities.values():
                if query.lower() in e.name.lower() or e.name.lower() in query.lower():
                    candidates.append(e)
            entity = candidates[0] if candidates else None

        if not entity:
            return {"strategy": "graph", "results": [], "total": 0}

        # 多跳查询
        hop_result = self.reasoner.multi_hop_query(entity.eid, hops=2)
        return {
            "strategy": "graph",
            "seed_entity": entity.name,
            "results": hop_result["results"][:top_k],
            "total": len(hop_result["results"]),
        }

    def _hybrid_retrieve(self, query: str, query_vec: List[float],
                         top_k: int) -> Dict:
        # 1. 向量检索找种子
        vec_results = self.vector_index.search(query_vec, top_k)

        # 2. 图谱扩展
        graph_expansions = []
        for r in vec_results[:3]:  # 取前3个种子
            entity = self.graph.get_entity(r["id"])
            if entity:
                neighbors = self.graph.get_neighbors(entity.eid)
                for nid in neighbors[:3]:
                    n_entity = self.graph.get_entity(nid)
                    if n_entity:
                        graph_expansions.append({
                            "id": nid,
                            "entity": n_entity.name,
                            "source": entity.name,
                            "score": r["score"] * 0.8,
                        })

        # 3. 合并去重
        seen = set()
        merged = []
        for r in vec_results:
            if r["id"] not in seen:
                seen.add(r["id"])
                merged.append(r)
        for r in graph_expansions:
            if r["id"] not in seen:
                seen.add(r["id"])
                merged.append(r)

        merged.sort(key=lambda x: -x.get("score", 0))
        return {
            "strategy": "hybrid",
            "results": merged[:top_k],
            "total": len(merged),
        }

    @staticmethod
    def _default_embed(text: str) -> List[float]:
        h = hashlib.md5(text.encode()).digest()
        vec = [(b / 255.0 - 0.5) * 2 for b in h[:16]]
        while len(vec) < 128:
            vec.append(random.gauss(0, 0.1))
        return vec[:128]


# ============================================================
# 上下文构建器
# ============================================================

class ContextBuilder:
    """上下文构建器 — 将检索结果组装为模型输入"""

    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens

    def build(self, query: str, retrieval_result: Dict,
              graph: Optional[KnowledgeGraph] = None) -> Dict:
        """构建增强上下文"""
        context_parts = []

        # 1. 检索结果
        for r in retrieval_result.get("results", []):
            if "entity" in r:
                context_parts.append(f"[知识] {r['entity']}")
            elif "text" in r:
                context_parts.append(f"[文档] {r['text']}")
            else:
                context_parts.append(f"[相关] {r.get('id', 'unknown')}")

        # 2. 图谱信息
        if graph:
            for r in retrieval_result.get("results", []):
                eid = r.get("id") or r.get("eid")
                if eid and eid in graph.entities:
                    entity = graph.entities[eid]
                    rels = graph.get_relations_of(eid)
                    for rel in rels[:3]:
                        head = graph.get_entity(rel.head)
                        tail = graph.get_entity(rel.tail)
                        if head and tail:
                            context_parts.append(
                                f"[三元组] {head.name} --{rel.relation}--> {tail.name}")

        context_text = "\n".join(context_parts[:20])  # 限制上下文长度

        return {
            "query": query,
            "context": context_text,
            "context_length": len(context_text),
            "num_sources": len(context_parts),
            "retrieval_strategy": retrieval_result.get("strategy", "unknown"),
        }


# ============================================================
# RAG引擎
# ============================================================

class RAGEngine:
    """RAG引擎 — 检索增强生成

    流程: query -> 检索知识 -> 构建上下文 -> 生成回答
    """

    def __init__(self, graph: Optional[KnowledgeGraph] = None,
                 vector_index: Optional[VectorIndex] = None):
        self.graph = graph or KnowledgeGraph()
        self.vector_index = vector_index or VectorIndex(dim=128)
        self.retriever = KnowledgeRetriever(self.graph, self.vector_index)
        self.context_builder = ContextBuilder()

    def add_knowledge(self, text: str, entity_name: Optional[str] = None,
                      entity_type: EntityType = EntityType.CONCEPT) -> None:
        """添加知识"""
        # 添加到向量索引
        vec = self.retriever.embed_fn(text)
        cid = f"chunk_{self.vector_index.size()}"
        self.vector_index.add(cid, vec)

        # 如果有实体名, 添加到图谱
        if entity_name:
            eid = f"entity_{len(self.graph.entities)}"
            entity = Entity(eid=eid, name=entity_name, entity_type=entity_type)
            self.graph.add_entity(entity)

    def answer(self, query: str, strategy: RetrievalStrategy = RetrievalStrategy.HYBRID) -> Dict:
        """RAG回答"""
        # 1. 检索
        retrieval = self.retriever.retrieve(query, top_k=5, strategy=strategy)

        # 2. 构建上下文
        context = self.context_builder.build(query, retrieval, self.graph)

        # 3. 模拟生成 (实际调用LLM)
        answer = self._generate(query, context)

        return {
            "query": query,
            "answer": answer,
            "context": context["context"],
            "sources": retrieval["results"],
            "strategy": retrieval["strategy"],
        }

    def _generate(self, query: str, context: Dict) -> str:
        """模拟生成回答"""
        if context["num_sources"] == 0:
            return f"关于'{query}', 我目前没有找到相关知识。"
        return f"基于{context['num_sources']}条知识, 关于'{query}'的回答: [检索到的上下文已注入]"

    def get_stats(self) -> Dict:
        return {
            "graph_stats": self.graph.get_stats(),
            "vector_index_size": self.vector_index.size(),
        }


# ============================================================
# 实体链接器
# ============================================================

class EntityLinker:
    """实体链接器 — 将文本中的实体链接到知识图谱"""

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self._build_index()

    def _build_index(self):
        self._surface_forms = {}
        for entity in self.graph.entities.values():
            self._surface_forms[entity.name] = entity.eid
            for alias in entity.aliases:
                self._surface_forms[alias] = entity.eid

    def link(self, text: str) -> List[Dict]:
        """链接文本中的实体"""
        linked = []
        for surface, eid in self._surface_forms.items():
            if surface in text:
                entity = self.graph.get_entity(eid)
                if entity:
                    linked.append({
                        "text": surface,
                        "eid": eid,
                        "entity": entity.name,
                        "type": entity.entity_type.value,
                        "start": text.index(surface),
                        "end": text.index(surface) + len(surface),
                    })
        linked.sort(key=lambda x: x["start"])
        return linked


# ============================================================
# 事实检查器
# ============================================================

class FactChecker:
    """事实检查器 — 基于知识图谱验证陈述"""

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.reasoner = GraphReasoner(graph)

    def check(self, statement: str) -> Dict:
        """检查陈述的事实性

        Returns:
            {verdict: true/false/uncertain, evidence: [...], confidence: float}
        """
        # 查找相关实体
        entities = []
        for entity in self.graph.entities.values():
            if entity.name in statement:
                entities.append(entity)

        if not entities:
            return {
                "statement": statement,
                "verdict": "uncertain",
                "evidence": [],
                "confidence": 0.0,
                "reason": "no_entities_found",
            }

        # 查找实体间关系
        evidence = []
        confidence = 0.0
        verdict = "uncertain"

        if len(entities) >= 2:
            head, tail = entities[0], entities[1]

            # 优先检查直接关系 (准确率最高)
            rels = self.graph.query_triples(head=head.eid, tail=tail.eid)
            if rels:
                evidence.append({
                    "head": head.name,
                    "tail": tail.name,
                    "relation": rels[0].relation,
                })
                confidence = rels[0].confidence
                verdict = "true"
            else:
                # 检查反向关系 (陈述为假)
                rels = self.graph.query_triples(head=tail.eid, tail=head.eid)
                if rels:
                    verdict = "false"
                    confidence = rels[0].confidence
                    evidence.append({
                        "head": tail.name,
                        "tail": head.name,
                        "relation": rels[0].relation,
                    })
                else:
                    # 无直接关系, 尝试推理路径
                    inferences = self.reasoner.infer_relation(head.eid, tail.eid)
                    if inferences:
                        evidence.append({
                            "head": head.name,
                            "tail": tail.name,
                            "paths": [inf["relations"] for inf in inferences[:3]],
                        })
                        confidence = min(1.0, len(inferences) * 0.3)
                        verdict = "likely_true" if confidence > 0.5 else "uncertain"

        return {
            "statement": statement,
            "verdict": verdict,
            "evidence": evidence,
            "confidence": round(confidence, 3),
            "entities_found": [e.name for e in entities],
            "reason": "verified" if verdict != "uncertain" else "insufficient_evidence",
        }


# ============================================================
# 知识更新器
# ============================================================

class KnowledgeUpdater:
    """知识更新器 — 增量更新知识图谱"""

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.update_log: List[Dict] = []

    def add_triple(self, head_name: str, relation: str, tail_name: str,
                   entity_type: EntityType = EntityType.CONCEPT) -> bool:
        """添加三元组"""
        # 获取或创建头实体
        head = self.graph.find_entity_by_name(head_name)
        if not head:
            head = Entity(eid=f"entity_{len(self.graph.entities)}",
                          name=head_name, entity_type=entity_type)
            self.graph.add_entity(head)

        # 获取或创建尾实体
        tail = self.graph.find_entity_by_name(tail_name)
        if not tail:
            tail = Entity(eid=f"entity_{len(self.graph.entities)}",
                          name=tail_name, entity_type=entity_type)
            self.graph.add_entity(tail)

        # 添加关系
        rel = Relation(rid=f"rel_{len(self.graph.relations)}",
                       head=head.eid, relation=relation, tail=tail.eid)
        success = self.graph.add_relation(rel)

        if success:
            self.update_log.append({
                "action": "add_triple",
                "head": head_name, "relation": relation, "tail": tail_name,
                "timestamp": time.time(),
            })

        return success

    def merge_entities(self, eid1: str, eid2: str) -> bool:
        """合并两个实体"""
        e1 = self.graph.get_entity(eid1)
        e2 = self.graph.get_entity(eid2)
        if not e1 or not e2:
            return False

        # 将e2的别名合并到e1
        if e2.name not in e1.aliases:
            e1.aliases.append(e2.name)
        e1.aliases.extend(e2.aliases)

        # 将e2的关系转移到e1
        for rel in self.graph.relations:
            if rel.head == eid2:
                rel.head = eid1
            if rel.tail == eid2:
                rel.tail = eid1

        # 删除e2
        del self.graph.entities[eid2]
        self.update_log.append({"action": "merge", "from": eid2, "to": eid1})
        return True

    def get_update_count(self) -> int:
        return len(self.update_log)


# ============================================================
# 图谱嵌入器 (TransE)
# ============================================================

class GraphEmbedder:
    """图谱嵌入器 — TransE算法

    将实体和关系映射到向量空间
    目标: h + r ≈ t (头实体向量 + 关系向量 ≈ 尾实体向量)
    """

    def __init__(self, graph: KnowledgeGraph, dim: int = 64):
        self.graph = graph
        self.dim = dim
        self.entity_embeddings: Dict[str, List[float]] = {}
        self.relation_embeddings: Dict[str, List[float]] = {}
        self._init_embeddings()

    def _init_embeddings(self):
        for eid in self.graph.entities:
            self.entity_embeddings[eid] = [random.gauss(0, 0.1) for _ in range(self.dim)]
        for rel in self.graph.relations:
            if rel.relation not in self.relation_embeddings:
                self.relation_embeddings[rel.relation] = [random.gauss(0, 0.1) for _ in range(self.dim)]

    def train(self, epochs: int = 50, lr: float = 0.01) -> List[float]:
        """训练TransE嵌入"""
        losses = []
        triples = [(r.head, r.relation, r.tail) for r in self.graph.relations]

        if not triples:
            return losses

        for epoch in range(epochs):
            total_loss = 0.0
            for h, r, t in triples:
                if h not in self.entity_embeddings or t not in self.entity_embeddings:
                    continue
                if r not in self.relation_embeddings:
                    self.relation_embeddings[r] = [random.gauss(0, 0.1) for _ in range(self.dim)]

                # h + r - t
                h_vec = self.entity_embeddings[h]
                r_vec = self.relation_embeddings[r]
                t_vec = self.entity_embeddings[t]

                # 损失 = ||h + r - t||
                diff = [h_vec[i] + r_vec[i] - t_vec[i] for i in range(self.dim)]
                loss = math.sqrt(sum(d * d for d in diff))
                total_loss += loss

                # 梯度下降 (简化)
                for i in range(self.dim):
                    grad = diff[i] / max(loss, 1e-8)
                    h_vec[i] -= lr * grad
                    r_vec[i] -= lr * grad
                    t_vec[i] += lr * grad

                # L2归一化
                self._normalize(h_vec)
                self._normalize(t_vec)

            avg_loss = total_loss / len(triples)
            losses.append(avg_loss)

        return losses

    def predict_tail(self, head: str, relation: str, top_k: int = 5) -> List[Dict]:
        """预测最可能的尾实体"""
        if head not in self.entity_embeddings or relation not in self.relation_embeddings:
            return []

        h_vec = self.entity_embeddings[head]
        r_vec = self.relation_embeddings[relation]
        target = [h_vec[i] + r_vec[i] for i in range(self.dim)]

        scores = []
        for eid, e_vec in self.entity_embeddings.items():
            if eid == head:
                continue
            dist = math.sqrt(sum((target[i] - e_vec[i]) ** 2 for i in range(self.dim)))
            scores.append({"eid": eid, "distance": round(dist, 4)})

        scores.sort(key=lambda x: x["distance"])
        return scores[:top_k]

    @staticmethod
    def _normalize(vec: List[float]) -> None:
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            for i in range(len(vec)):
                vec[i] /= norm


# ============================================================
# 子图提取器
# ============================================================

class SubgraphExtractor:
    """子图提取器 — 提取相关的子图"""

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    def extract_by_entities(self, eids: List[str], hops: int = 1) -> Dict:
        """基于实体集合提取子图"""
        all_entities = set(eids)
        all_relations = []

        for eid in eids:
            sub = self.graph.extract_subgraph(eid, depth=hops)
            for e in sub["entities"]:
                all_entities.add(e.get("eid", ""))
            all_relations.extend(sub["relations"])

        # 去重
        seen_rels = set()
        unique_rels = []
        for r in all_relations:
            key = (r["head"], r["relation"], r["tail"])
            if key not in seen_rels:
                seen_rels.add(key)
                unique_rels.append(r)

        return {
            "entities": [self.graph.entities[eid].to_dict()
                         for eid in all_entities if eid in self.graph.entities],
            "relations": unique_rels,
            "num_entities": len(all_entities),
            "num_relations": len(unique_rels),
        }

    def extract_by_relation_type(self, rel_type: str) -> Dict:
        """基于关系类型提取子图"""
        rels = [r for r in self.graph.relations if r.relation == rel_type]
        eids = set()
        for r in rels:
            eids.add(r.head)
            eids.add(r.tail)

        return {
            "relation_type": rel_type,
            "entities": [self.graph.entities[eid].to_dict()
                         for eid in eids if eid in self.graph.entities],
            "relations": [r.to_dict() for r in rels],
            "num_entities": len(eids),
            "num_relations": len(rels),
        }


# ============================================================
# 多跳推理器
# ============================================================

class MultiHopReasoner:
    """多跳推理器 — 支持复杂的多跳问答"""

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.reasoner = GraphReasoner(graph)

    def answer_multi_hop(self, question: str, entities: List[str],
                         hops: int = 2) -> Dict:
        """多跳问答

        Args:
            question: 问题
            entities: 问题中识别到的实体名
            hops: 推理跳数

        Returns:
            推理链和答案
        """
        chains = []
        for entity_name in entities:
            entity = self.graph.find_entity_by_name(entity_name)
            if not entity:
                continue

            hop_result = self.reasoner.multi_hop_query(entity.eid, hops=hops)
            chain = {
                "start_entity": entity_name,
                "hops": [],
            }

            for r in hop_result["results"]:
                target = self.graph.get_entity(r["eid"])
                chain["hops"].append({
                    "entity": target.name if target else r["eid"],
                    "distance": r["distance"],
                })

            chains.append(chain)

        return {
            "question": question,
            "reasoning_chains": chains,
            "total_entities_explored": sum(len(c["hops"]) for c in chains),
        }


# ============================================================
# 知识流水线
# ============================================================

class KnowledgePipeline:
    """知识流水线 — 端到端知识管理"""

    def __init__(self):
        self.graph = KnowledgeGraph()
        self.vector_index = VectorIndex(dim=128)
        self.rag = RAGEngine(self.graph, self.vector_index)
        self.linker = EntityLinker(self.graph)
        self.fact_checker = FactChecker(self.graph)
        self.updater = KnowledgeUpdater(self.graph)
        self.embedder = GraphEmbedder(self.graph)
        self.extractor = SubgraphExtractor(self.graph)
        self.multi_hop = MultiHopReasoner(self.graph)

    def ingest(self, text: str, entities: Optional[List[str]] = None,
               triples: Optional[List[Tuple[str, str, str]]] = None) -> Dict:
        """摄入知识"""
        # 添加文本到向量索引
        self.rag.add_knowledge(text)

        # 添加实体
        added_entities = 0
        if entities:
            for name in entities:
                if not self.graph.find_entity_by_name(name):
                    eid = f"entity_{len(self.graph.entities)}"
                    self.graph.add_entity(Entity(eid=eid, name=name))
                    added_entities += 1

        # 添加三元组
        added_triples = 0
        if triples:
            for h, r, t in triples:
                if self.updater.add_triple(h, r, t):
                    added_triples += 1

        return {
            "text_added": True,
            "entities_added": added_entities,
            "triples_added": added_triples,
        }

    def query(self, question: str) -> Dict:
        """查询知识"""
        # RAG回答
        rag_result = self.rag.answer(question)

        # 实体链接
        linked = self.linker.link(question)

        # 多跳推理
        entity_names = [l["entity"] for l in linked]
        if not entity_names:
            entity_names = [e.name for e in list(self.graph.entities.values())[:2]]

        hop_result = self.multi_hop.answer_multi_hop(question, entity_names, hops=2)

        return {
            "rag_answer": rag_result,
            "linked_entities": linked,
            "multi_hop": hop_result,
            "graph_stats": self.graph.get_stats(),
        }

    def verify(self, statement: str) -> Dict:
        """验证陈述"""
        return self.fact_checker.check(statement)

    def get_pipeline_info(self) -> Dict:
        return {
            "pipeline_type": "KnowledgePipeline",
            "graph_stats": self.graph.get_stats(),
            "vector_index_size": self.vector_index.size(),
            "components": [
                "KnowledgeGraph", "GraphReasoner", "VectorIndex",
                "KnowledgeRetriever", "ContextBuilder", "RAGEngine",
                "EntityLinker", "FactChecker", "KnowledgeUpdater",
                "GraphEmbedder", "SubgraphExtractor", "MultiHopReasoner",
            ],
        }


# ============================================================
# 测试函数
# ============================================================

def _test_knowledge_graph():
    print("  [测试] KnowledgeGraph...")
    kg = KnowledgeGraph()
    e1 = Entity(eid="e1", name="Python", entity_type=EntityType.CONCEPT)
    e2 = Entity(eid="e2", name="Guido", entity_type=EntityType.PERSON)
    kg.add_entity(e1)
    kg.add_entity(e2)
    kg.add_relation(Relation(rid="r1", head="e2", relation="created_by", tail="e1"))

    assert kg.get_entity("e1").name == "Python"
    assert kg.find_entity_by_name("Python") is not None
    assert "e1" in kg.get_neighbors("e2")
    assert len(kg.query_triples(head="e2")) == 1

    sub = kg.extract_subgraph("e2", depth=1)
    assert len(sub["entities"]) == 2
    print("    PASS")


def _test_graph_reasoner():
    print("  [测试] GraphReasoner...")
    kg = KnowledgeGraph()
    for i in range(5):
        kg.add_entity(Entity(eid=f"e{i}", name=f"Entity{i}"))
    kg.add_relation(Relation(rid="r0", head="e0", relation="r", tail="e1"))
    kg.add_relation(Relation(rid="r1", head="e1", relation="r", tail="e2"))
    kg.add_relation(Relation(rid="r2", head="e2", relation="r", tail="e3"))

    reasoner = GraphReasoner(kg)
    paths = reasoner.find_paths("e0", "e3", max_depth=4)
    assert len(paths) > 0

    hop = reasoner.multi_hop_query("e0", hops=2)
    assert len(hop["results"]) > 0
    print("    PASS")


def _test_vector_index():
    print("  [测试] VectorIndex...")
    idx = VectorIndex(dim=16)
    for i in range(10):
        vec = [random.gauss(0, 1) for _ in range(16)]
        idx.add(f"item_{i}", vec)

    query = [random.gauss(0, 1) for _ in range(16)]
    results = idx.search(query, top_k=3)
    assert len(results) == 3
    assert results[0]["score"] >= results[1]["score"]
    print("    PASS")


def _test_knowledge_retriever():
    print("  [测试] KnowledgeRetriever...")
    kg = KnowledgeGraph()
    kg.add_entity(Entity(eid="e1", name="AI", entity_type=EntityType.CONCEPT))
    kg.add_entity(Entity(eid="e2", name="ML", entity_type=EntityType.CONCEPT))
    kg.add_relation(Relation(rid="r1", head="e2", relation="is_a", tail="e1"))

    vi = VectorIndex(dim=128)
    vi.add("e1", [random.gauss(0, 1) for _ in range(128)])
    vi.add("e2", [random.gauss(0, 1) for _ in range(128)])

    retriever = KnowledgeRetriever(kg, vi)
    result = retriever.retrieve("AI", strategy=RetrievalStrategy.VECTOR)
    assert result["total"] > 0

    result = retriever.retrieve("AI", strategy=RetrievalStrategy.GRAPH)
    assert "seed_entity" in result

    result = retriever.retrieve("AI", strategy=RetrievalStrategy.HYBRID)
    assert result["total"] > 0
    print("    PASS")


def _test_context_builder():
    print("  [测试] ContextBuilder...")
    cb = ContextBuilder(max_tokens=256)
    retrieval = {
        "strategy": "hybrid",
        "results": [{"id": "e1", "entity": "AI", "score": 0.9}],
    }
    kg = KnowledgeGraph()
    kg.add_entity(Entity(eid="e1", name="AI"))
    kg.add_entity(Entity(eid="e2", name="ML"))
    kg.add_relation(Relation(rid="r1", head="e2", relation="is_a", tail="e1"))

    ctx = cb.build("什么是AI?", retrieval, kg)
    assert "context" in ctx
    assert ctx["num_sources"] > 0
    print("    PASS")


def _test_rag_engine():
    print("  [测试] RAGEngine...")
    rag = RAGEngine()
    rag.add_knowledge("Python是Guido创建的编程语言", "Python", EntityType.CONCEPT)
    rag.add_knowledge("机器学习是AI的子领域", "ML", EntityType.CONCEPT)

    result = rag.answer("Python", strategy=RetrievalStrategy.VECTOR)
    assert "answer" in result
    assert "context" in result
    print("    PASS")


def _test_entity_linker():
    print("  [测试] EntityLinker...")
    kg = KnowledgeGraph()
    kg.add_entity(Entity(eid="e1", name="Python", aliases=["Python语言"]))
    linker = EntityLinker(kg)

    linked = linker.link("Python是一种编程语言")
    assert len(linked) > 0
    assert linked[0]["entity"] == "Python"

    linked2 = linker.link("Python语言很好用")
    assert len(linked2) > 0
    print("    PASS")


def _test_fact_checker():
    print("  [测试] FactChecker...")
    kg = KnowledgeGraph()
    kg.add_entity(Entity(eid="e1", name="Python"))
    kg.add_entity(Entity(eid="e2", name="Guido"))
    kg.add_relation(Relation(rid="r1", head="e1", relation="created_by", tail="e2"))

    checker = FactChecker(kg)
    result = checker.check("Python created_by Guido")
    assert "verdict" in result
    assert result["verdict"] != "uncertain" or result.get("reason", "") == "no_entities_found"
    print("    PASS")


def _test_knowledge_updater():
    print("  [测试] KnowledgeUpdater...")
    kg = KnowledgeGraph()
    updater = KnowledgeUpdater(kg)

    assert updater.add_triple("AI", "includes", "ML")
    assert updater.add_triple("ML", "includes", "DL")
    assert kg.find_entity_by_name("AI") is not None
    assert len(kg.relations) == 2
    assert updater.get_update_count() == 2
    print("    PASS")


def _test_graph_embedder():
    print("  [测试] GraphEmbedder...")
    kg = KnowledgeGraph()
    for i in range(5):
        kg.add_entity(Entity(eid=f"e{i}", name=f"Entity{i}"))
    kg.add_relation(Relation(rid="r0", head="e0", relation="r1", tail="e1"))
    kg.add_relation(Relation(rid="r1", head="e1", relation="r1", tail="e2"))
    kg.add_relation(Relation(rid="r2", head="e0", relation="r2", tail="e3"))

    embedder = GraphEmbedder(kg, dim=32)
    losses = embedder.train(epochs=20, lr=0.01)
    assert len(losses) == 20
    assert losses[-1] <= losses[0]  # 损失应该下降

    preds = embedder.predict_tail("e0", "r1", top_k=3)
    assert len(preds) <= 3
    print("    PASS")


def _test_subgraph_extractor():
    print("  [测试] SubgraphExtractor...")
    kg = KnowledgeGraph()
    for i in range(6):
        kg.add_entity(Entity(eid=f"e{i}", name=f"Entity{i}"))
    kg.add_relation(Relation(rid="r0", head="e0", relation="r", tail="e1"))
    kg.add_relation(Relation(rid="r1", head="e1", relation="r", tail="e2"))
    kg.add_relation(Relation(rid="r2", head="e2", relation="r", tail="e3"))

    ext = SubgraphExtractor(kg)
    sub = ext.extract_by_entities(["e0", "e1"], hops=1)
    assert sub["num_entities"] > 0
    assert sub["num_relations"] > 0

    sub2 = ext.extract_by_relation_type("r")
    assert sub2["num_relations"] == 3
    print("    PASS")


def _test_multi_hop_reasoner():
    print("  [测试] MultiHopReasoner...")
    kg = KnowledgeGraph()
    for i in range(5):
        kg.add_entity(Entity(eid=f"e{i}", name=f"Entity{i}"))
    kg.add_relation(Relation(rid="r0", head="e0", relation="r", tail="e1"))
    kg.add_relation(Relation(rid="r1", head="e1", relation="r", tail="e2"))
    kg.add_relation(Relation(rid="r2", head="e2", relation="r", tail="e3"))

    reasoner = MultiHopReasoner(kg)
    result = reasoner.answer_multi_hop("查询", ["Entity0"], hops=2)
    assert len(result["reasoning_chains"]) > 0
    assert result["total_entities_explored"] > 0
    print("    PASS")


def _test_pipeline():
    print("  [测试] KnowledgePipeline...")
    pipeline = KnowledgePipeline()

    # 摄入知识
    result = pipeline.ingest(
        "Python是Guido创建的编程语言",
        entities=["Python", "Guido"],
        triples=[("Python", "created_by", "Guido"),
                 ("Python", "is_a", "编程语言")]
    )
    assert result["entities_added"] == 2
    assert result["triples_added"] == 2

    # 查询
    query_result = pipeline.query("Python")
    assert "rag_answer" in query_result
    assert "graph_stats" in query_result

    # 验证
    verify_result = pipeline.verify("Python created_by Guido")
    assert "verdict" in verify_result

    info = pipeline.get_pipeline_info()
    assert "components" in info
    print("    PASS")


def _test_integration():
    print("  [测试] 集成测试: 完整知识图谱系统...")
    pipeline = KnowledgePipeline()

    # 批量摄入
    triples = [
        ("AI", "includes", "ML"),
        ("ML", "includes", "DL"),
        ("DL", "uses", "NeuralNetwork"),
        ("NeuralNetwork", "inspired_by", "Brain"),
        ("Python", "used_for", "ML"),
        ("Python", "created_by", "Guido"),
    ]
    for h, r, t in triples:
        pipeline.ingest(f"{h} {r} {t}", entities=[h, t], triples=[(h, r, t)])

    # 检索
    result = pipeline.query("Python和ML的关系")
    assert result["graph_stats"]["num_entities"] > 0

    # 多跳推理
    hop_result = pipeline.multi_hop.answer_multi_hop("AI包含什么", ["AI"], hops=2)
    assert len(hop_result["reasoning_chains"]) > 0

    # 图谱嵌入
    losses = pipeline.embedder.train(epochs=10)
    assert len(losses) == 10

    # 子图提取
    ai_entity = pipeline.graph.find_entity_by_name("AI")
    if ai_entity:
        sub = pipeline.extractor.extract_by_entities([ai_entity.eid], hops=2)
        assert sub["num_entities"] > 0

    print("    PASS")


# ============================================================
# 主入口
# ============================================================

def main():
    print()
    print("=" * 70)
    print("  灵元模型 - 知识图谱增强系统模块 (Part 25) 自测")
    print("=" * 70)
    print()

    tests = [
        ("KnowledgeGraph", _test_knowledge_graph),
        ("GraphReasoner", _test_graph_reasoner),
        ("VectorIndex", _test_vector_index),
        ("KnowledgeRetriever", _test_knowledge_retriever),
        ("ContextBuilder", _test_context_builder),
        ("RAGEngine", _test_rag_engine),
        ("EntityLinker", _test_entity_linker),
        ("FactChecker", _test_fact_checker),
        ("KnowledgeUpdater", _test_knowledge_updater),
        ("GraphEmbedder", _test_graph_embedder),
        ("SubgraphExtractor", _test_subgraph_extractor),
        ("MultiHopReasoner", _test_multi_hop_reasoner),
        ("KnowledgePipeline", _test_pipeline),
        ("Integration", _test_integration),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  [FAIL] {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"  自测结果: {passed} 通过, {failed} 失败, 共 {len(tests)} 项")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
    else:
        print("  所有测试通过!")


if __name__ == "__main__":
    main()
