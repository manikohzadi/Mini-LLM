from word2vec import Word2Vec, PersianPreprocessor

pre = PersianPreprocessor()
corpus = pre.preprocess_corpus(["your text here king mani you are very man", "fatemeh you are a woman"])

model = Word2Vec(vector_size=100, epochs=10)
model.fit(corpus)

vec = model.get_vector("king")              # بردار یک کلمه
model.most_similar("king", topn=10)         # شبیه‌ترین‌ها
model.analogy("man", "king", "woman")       # قیاس: king - man + woman
model.save("my_vectors.txt")                # فرمت word2vec (سازگار با gensim)