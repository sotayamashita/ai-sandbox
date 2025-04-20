import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class ConceptEmbedder(nn.Module):
    """
    KG情報を圧縮して「コンセプトベクトル」を生成するためのエンベッダー
    """
    def __init__(self, node_dim, hidden_dim, output_dim):
        super(ConceptEmbedder, self).__init__()
        self.fc1 = nn.Linear(node_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, node_features):
        """
        :param node_features: ノード特徴量 (batch_size, node_dim)
        :return: コンセプトベクトル (batch_size, output_dim)
        """
        x = F.gelu(self.fc1(node_features))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ConceptFormer:
    """
    ConceptFormerのメインクラス
    KGの情報を圧縮してLLMの入力層に直接注入します
    """
    def __init__(self, model_name='gpt2', device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # GPT-2モデルとトークナイザーをロード
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # モデルの埋め込み次元
        self.embedding_dim = self.model.config.n_embd
        
        # コンセプトエンベッダー
        self.concept_embedder = ConceptEmbedder(
            node_dim=self.embedding_dim,  # 入力はKGエンベディングと同じ次元
            hidden_dim=self.embedding_dim*2,
            output_dim=self.embedding_dim  # 出力はGPT-2の埋め込み次元と同じ
        ).to(self.device)
        
        # ルックアップテーブル（KGノード → コンセプトベクトル）
        self.concept_lookup = {}
    
    def train_concept_embedder(self, kg_embeddings, labels, epochs=10, lr=2e-5, batch_size=32):
        """
        コンセプトエンベッダーをトレーニングする
        
        :param kg_embeddings: KGエンベディングのリスト
        :param labels: 対応するラベル
        """
        optimizer = torch.optim.AdamW(self.concept_embedder.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(kg_embeddings, dtype=torch.float32).to(self.device),
            torch.tensor(labels, dtype=torch.long).to(self.device)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.concept_embedder.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_embeddings, batch_labels in dataloader:
                optimizer.zero_grad()
                
                # コンセプトベクトルを生成
                concept_vectors = self.concept_embedder(batch_embeddings)
                
                # 元のGPT-2埋め込み行列で予測を生成
                word_embeddings = self.model.transformer.wte.weight
                similarity = torch.matmul(concept_vectors, word_embeddings.t())
                
                # 損失計算と最適化
                loss = criterion(similarity, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")
    
    def generate_concept_lookup(self, kg_nodes, kg_embeddings):
        """
        KGノードからコンセプトベクトルへのルックアップテーブルを生成
        
        :param kg_nodes: KGノードIDのリスト
        :param kg_embeddings: 対応するKGエンベディングのリスト
        """
        self.concept_embedder.eval()
        with torch.no_grad():
            for node_id, embedding in zip(kg_nodes, kg_embeddings):
                tensor_embedding = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
                concept_vector = self.concept_embedder(tensor_embedding).squeeze(0).cpu().numpy()
                self.concept_lookup[node_id] = concept_vector
    
    def inject_concept_vector(self, input_text, node_id):
        """
        テキスト入力にコンセプトベクトルを注入
        
        :param input_text: 入力テキスト
        :param node_id: 注入するKGノードのID
        :return: モデル出力
        """
        if node_id not in self.concept_lookup:
            raise ValueError(f"Node ID {node_id} not found in concept lookup table")
            
        # テキストをトークン化
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        # トークン埋め込みを取得
        with torch.no_grad():
            token_embeddings = self.model.transformer.wte(input_ids)
        
        # コンセプトベクトルを注入
        concept_vector = torch.tensor(self.concept_lookup[node_id], dtype=torch.float32).to(self.device)
        modified_embeddings = torch.cat([
            concept_vector.unsqueeze(0).unsqueeze(0),  # [1, 1, embedding_dim]
            token_embeddings  # [1, seq_len, embedding_dim]
        ], dim=1)
        
        # モデルの残りの部分に通す
        outputs = self.model(inputs_embeds=modified_embeddings)
        
        return outputs
    
    def generate_text_with_concept(self, prompt, node_id, max_length=50):
        """
        コンセプトベクトルを注入してテキスト生成
        
        :param prompt: 入力プロンプト
        :param node_id: 注入するKGノードのID
        :param max_length: 生成する最大トークン数
        :return: 生成されたテキスト
        """
        if node_id not in self.concept_lookup:
            raise ValueError(f"Node ID {node_id} not found in concept lookup table")
            
        # プロンプトをトークン化
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # トークン埋め込みを取得
        with torch.no_grad():
            token_embeddings = self.model.transformer.wte(input_ids)
        
        # コンセプトベクトルを注入
        concept_vector = torch.tensor(self.concept_lookup[node_id], dtype=torch.float32).to(self.device)
        modified_embeddings = torch.cat([
            concept_vector.unsqueeze(0).unsqueeze(0),  # [1, 1, embedding_dim]
            token_embeddings  # [1, seq_len, embedding_dim]
        ], dim=1)
        
        # 位置エンコーディングのオフセットを設定（コンセプトベクトルの分）
        position_ids = torch.arange(modified_embeddings.size(1), dtype=torch.long, device=self.device).unsqueeze(0)
        attention_mask = torch.ones(modified_embeddings.size()[:2], device=self.device)
        
        # 生成
        output_ids = self.model.generate(
            inputs_embeds=modified_embeddings,
            position_ids=position_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # 出力をデコード（最初のコンセプトベクトルに対応するトークンを除く）
        generated_text = self.tokenizer.decode(output_ids[0][1:], skip_special_tokens=True)
        
        return generated_text 
