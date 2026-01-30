import json
from tokenizer.tokenizer import tokenize
from tokenizer.build_vocab import build_vocab
from embedding.word2vec import Word2Vec


WINDOW_SIZE = 2
EMBEDDING_DIM = 128
NEGATIVE_SAMPLES = 5
EPOCHS = 1


def generate_pairs(token_ids):
    pairs = []
    for i, center in enumerate(token_ids):
        start = max(0, i - WINDOW_SIZE)
        end = min(len(token_ids), i + WINDOW_SIZE + 1)

        for j in range(start, end):
            if i != j:
                pairs.append((center, token_ids[j]))
    return pairs


# --- آماده‌سازی داده ---
with open("data/sample_fa.txt", encoding="utf-8") as f:
    tokens = tokenize(f.read())

vocab = build_vocab([tokens])
token_to_id = vocab["token_to_id"]

token_ids = [token_to_id.get(t, token_to_id["<UNK>"]) for t in tokens]

# --- مدل ---
model = Word2Vec(
    vocab_size=len(token_to_id),
    embedding_dim=EMBEDDING_DIM,
    negative_samples=NEGATIVE_SAMPLES
)

pairs = generate_pairs(token_ids)

# --- آموزش ---
for epoch in range(EPOCHS):
    for center, context in pairs:
        model.train_pair(center, context)

# --- ذخیره ---
embeddings = {
    str(i): model.get_embedding(i)
    for i in range(len(token_to_id))
}

with open("word2vec_fa.json", "w", encoding="utf-8") as f:
    json.dump(embeddings, f)