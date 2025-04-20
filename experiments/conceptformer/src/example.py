import os
import json
import torch
import numpy as np
from pathlib import Path
from model import ConceptFormer
from kg_utils import KnowledgeGraph, prepare_training_data, prepare_lookup_data, prepare_synthetic_triples

# トイデータセットのセットアップ関数
def setup_toy_dataset():
    """
    テスト用のデータセットをセットアップ
    """
    # 出力ディレクトリ作成
    os.makedirs('data', exist_ok=True)
    
    # エンティティデータの作成
    # Q means Query or Quantum 
    entities = [
        {"id": "Q1", "label": "Albert Einstein"},
        {"id": "Q2", "label": "Physics"},
        {"id": "Q3", "label": "Nobel Prize"},
        {"id": "Q4", "label": "Germany"},
        {"id": "Q5", "label": "United States"},
    ]
    # JSONLフォーマットで書き込み（各エンティティを独立したJSON行として）
    with open('data/entities.jsonl', 'w', encoding='utf-8') as f:
        for entity in entities:
            f.write(json.dumps(entity) + '\n')
    
    # 関係データの作成
    # P means Property
    relations = [
        {"id": "P1", "label": "field of work"},
        {"id": "P2", "label": "award received"},
        {"id": "P3", "label": "country of citizenship"},
        {"id": "P4", "label": "educated at"},
    ]
    # JSONLフォーマットで書き込み（各関係を独立したJSON行として）
    with open('data/relations.jsonl', 'w', encoding='utf-8') as f:
        for relation in relations:
            f.write(json.dumps(relation) + '\n')
    
    # トリプルデータの作成
    triples = [
        "Q1 P1 Q2",  # Einstein - field of work - Physics
        "Q1 P2 Q3",  # Einstein - award received - Nobel Prize
        "Q1 P3 Q4",  # Einstein - country of citizenship - Germany
        "Q1 P3 Q5",  # Einstein - country of citizenship - USA
    ]
    Path('data/triples.txt').write_text('\n'.join(triples))
    
    # ランダムな埋め込みを生成（実際のプロジェクトでは既存の埋め込みを使用）
    embedding_size = 768  # GPT-2の埋め込みサイズ
    embeddings = {}
    
    for entity in entities:
        embeddings[entity["id"]] = np.random.randn(embedding_size).tolist()
    
    Path('data/embeddings.json').write_text(json.dumps(embeddings))
    
    return True

# メイン関数
def main():
    # トイデータセットのセットアップ
    setup_toy_dataset()
    
    # 知識グラフのロード
    kg = KnowledgeGraph()
    kg.load_entities('data/entities.jsonl')
    kg.load_relations('data/relations.jsonl')
    kg.load_triples('data/triples.txt')
    kg.load_entity_embeddings('data/embeddings.json')
    
    # ConceptFormerモデルの初期化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ConceptFormer(model_name='gpt2', device=device)
    
    # トレーニングデータの準備
    entity_ids = list(kg.entities.keys())
    kg_embeddings, labels = prepare_training_data(kg, entity_ids)
    
    print(f"Training data prepared: {len(kg_embeddings)} samples")
    
    # コンセプトエンベッダーのトレーニング
    print("Training concept embedder...")
    model.train_concept_embedder(
        kg_embeddings=kg_embeddings,
        labels=labels,
        epochs=5,
        lr=2e-5,
        batch_size=2
    )
    
    # ルックアップテーブルの生成
    print("Generating concept lookup table...")
    node_ids, kg_embeds = prepare_lookup_data(kg)
    model.generate_concept_lookup(node_ids, kg_embeds)
    
    # アインシュタインに関するサンプルプロンプト
    print("Generating text with injected concept vector...")
    
    einstein_id = "Q1"
    
    # プロンプトでアインシュタインに関するテキスト生成
    prompt = "The famous scientist "
    generated_text = model.generate_text_with_concept(prompt, einstein_id, max_length=30)
    print(f"\nPrompt: {prompt}")
    print(f"Generated (with concept vector): {generated_text}")
    
    # 合成トリプル文の生成
    print("\nGenerating synthetic triples...")
    template = "{subject} {relation} {object}."
    synthetic_sentences = prepare_synthetic_triples(kg, einstein_id, template)
    
    print(f"Synthetic sentences for {kg.get_entity_label(einstein_id)}:")
    for sentence in synthetic_sentences:
        print(f"- {sentence}")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 
