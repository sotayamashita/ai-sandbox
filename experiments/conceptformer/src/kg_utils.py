import os
import json
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional


class KnowledgeGraph:
    """
    シンプルな知識グラフのラッパークラス
    知識グラフは、エンティティ同士がどのような関係で繋がっているかを表す図
    """
    def __init__(self):
        self.entities = {}  # id -> entity
        self.relations = {}  # id -> relation
        self.triples = []  # [(head_id, relation_id, tail_id), ...]
        self.entity_embeddings = {}  # id -> embedding
        
    def load_entities(self, entity_file):
        """
        エンティティファイルを読み込む
        形式: {"id": "Q123", "label": "Albert Einstein", ...}
        """
        with open(entity_file, 'r', encoding='utf-8') as f:
            for line in f:
                entity = json.loads(line.strip())
                self.entities[entity['id']] = entity
    
    def load_relations(self, relation_file):
        """
        関係ファイルを読み込む
        形式: {"id": "P31", "label": "instance of", ...}
        """
        with open(relation_file, 'r', encoding='utf-8') as f:
            for line in f:
                relation = json.loads(line.strip())
                self.relations[relation['id']] = relation
    
    def load_triples(self, triples_file):
        """
        トリプルファイルを読み込む
        形式: head_id relation_id tail_id
        """
        with open(triples_file, 'r', encoding='utf-8') as f:
            for line in f:
                head, rel, tail = line.strip().split()
                self.triples.append((head, rel, tail))
    
    def load_entity_embeddings(self, embeddings_file):
        """
        エンティティ埋め込みファイルを読み込む
        形式: {entity_id: [emb_1, emb_2, ...], ...}
        """
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            self.entity_embeddings = json.load(f)
    
    def get_entity_neighborhood(self, entity_id, max_hops=1):
        """
        エンティティの近傍を取得する
        
        :param entity_id: エンティティID
        :param max_hops: 最大ホップ数
        :return: 近傍エンティティのリスト [(entity_id, relation_id, direction), ...]
        """
        neighborhood = []
        
        # 最初のホップ（直接繋がっているエンティティ）
        for head, rel, tail in self.triples:
            if head == entity_id:
                neighborhood.append((tail, rel, 'outgoing'))
            elif tail == entity_id:
                neighborhood.append((head, rel, 'incoming'))
        
        # max_hopsが1より大きい場合、追加のホップを処理
        if max_hops > 1:
            current_hop = 1
            current_entities = [n[0] for n in neighborhood]
            
            while current_hop < max_hops:
                next_entities = []
                
                for entity in current_entities:
                    for head, rel, tail in self.triples:
                        if head == entity and tail not in next_entities:
                            next_entities.append(tail)
                            if (tail, rel, 'outgoing') not in neighborhood:
                                neighborhood.append((tail, rel, 'outgoing'))
                        elif tail == entity and head not in next_entities:
                            next_entities.append(head)
                            if (head, rel, 'incoming') not in neighborhood:
                                neighborhood.append((head, rel, 'incoming'))
                
                current_entities = next_entities
                current_hop += 1
        
        return neighborhood
    
    def get_entity_embedding(self, entity_id):
        """
        エンティティの埋め込みを取得する
        
        :param entity_id: エンティティID
        :return: 埋め込みベクトル
        """
        if entity_id not in self.entity_embeddings:
            raise ValueError(f"Embedding for entity {entity_id} not found")
        
        return self.entity_embeddings[entity_id]
    
    def get_entity_label(self, entity_id):
        """
        エンティティのラベルを取得する
        
        :param entity_id: エンティティID
        :return: エンティティラベル
        """
        if entity_id not in self.entities:
            return entity_id
        
        return self.entities[entity_id].get('label', entity_id)
    
    def get_relation_label(self, relation_id):
        """
        関係のラベルを取得する
        
        :param relation_id: 関係ID
        :return: 関係ラベル
        """
        if relation_id not in self.relations:
            return relation_id
        
        return self.relations[relation_id].get('label', relation_id)


def prepare_training_data(kg: KnowledgeGraph, entity_ids: List[str]) -> Tuple[List[np.ndarray], List[int]]:
    """
    コンセプトエンベッダーのトレーニングデータを準備する
    
    :param kg: 知識グラフオブジェクト
    :param entity_ids: トレーニングに使用するエンティティIDのリスト
    :return: (KG埋め込みのリスト, エンティティIDのインデックスのリスト)
    """
    # ワードトゥインデックスのマッピングを作成
    entity_to_idx = {entity_id: idx for idx, entity_id in enumerate(entity_ids)}
    
    # トレーニングデータの準備
    kg_embeddings = []
    labels = []
    
    for entity_id in tqdm(entity_ids, desc="Preparing training data"):
        if entity_id in kg.entity_embeddings:
            kg_embeddings.append(kg.get_entity_embedding(entity_id))
            labels.append(entity_to_idx[entity_id])
    
    return kg_embeddings, labels


def prepare_lookup_data(kg: KnowledgeGraph, entity_ids: Optional[List[str]] = None) -> Tuple[List[str], List[np.ndarray]]:
    """
    ルックアップテーブル用のデータを準備する
    
    :param kg: 知識グラフオブジェクト
    :param entity_ids: 使用するエンティティIDのリスト（Noneの場合は全て）
    :return: (エンティティIDのリスト, KG埋め込みのリスト)
    """
    if entity_ids is None:
        entity_ids = list(kg.entity_embeddings.keys())
    
    # ルックアップデータの準備
    kg_embeddings = []
    valid_entity_ids = []
    
    for entity_id in tqdm(entity_ids, desc="Preparing lookup data"):
        if entity_id in kg.entity_embeddings:
            kg_embeddings.append(kg.get_entity_embedding(entity_id))
            valid_entity_ids.append(entity_id)
    
    return valid_entity_ids, kg_embeddings


def prepare_synthetic_triples(kg: KnowledgeGraph, entity_id: str, template: str) -> List[str]:
    """
    特定のエンティティに関する合成トリプル文を生成する
    
    :param kg: 知識グラフオブジェクト
    :param entity_id: エンティティID
    :param template: テンプレート文字列（{subject}、{relation}、{object}を含む）
    :return: 合成文のリスト
    """
    if entity_id not in kg.entities:
        raise ValueError(f"Entity {entity_id} not found in knowledge graph")
    
    entity_label = kg.get_entity_label(entity_id)
    sentences = []
    
    # エンティティが主語の場合のトリプルを探す
    for head, rel, tail in kg.triples:
        if head == entity_id:
            tail_label = kg.get_entity_label(tail)
            rel_label = kg.get_relation_label(rel)
            
            sentence = template.format(
                subject=entity_label,
                relation=rel_label,
                object=tail_label
            )
            sentences.append(sentence)
    
    # エンティティが目的語の場合のトリプルを探す
    for head, rel, tail in kg.triples:
        if tail == entity_id:
            head_label = kg.get_entity_label(head)
            rel_label = kg.get_relation_label(rel)
            
            sentence = template.format(
                subject=head_label,
                relation=rel_label,
                object=entity_label
            )
            sentences.append(sentence)
    
    return sentences 
