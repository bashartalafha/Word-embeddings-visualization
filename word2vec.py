from gensim.models import Word2Vec

sentences = [line.split() for line in open("All_data.txt")]
model = Word2Vec(sentences, sg=1, window=3, iter=25, size=128, min_count=0, workers=20)
model.save("All_data_vectors")

# lookup a vector
vector = model ['في']

# Find most similar words:
print(model.most_similar(['في']))

