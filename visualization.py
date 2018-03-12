import numpy as np
from six.moves import xrange  
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models.wrappers import FastText
import gensim
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib.patches as mpatches

# Visualize the embeddings.
def plot_with_labels(low_dim_embs, labels):
  filename='tsne.png'


  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure()  # in inches


  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y,color='b')
    plt.annotate(get_display(arabic_reshaper.reshape(label)),
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.show()
  plt.savefig(filename)

try:
 
  print ("Load all vocabs and embeddings ...")
  all_model = FastText()
  all_model= all_model.load_fasttext_format("All_data_vectors")		# Fasttext
#  all_model= gensim.models.KeyedVectors.load_word2vec_format('All_data_vectors.vec')        # Word2vec

  all_word_vectors= all_model.wv
  vocabs = all_word_vectors.vocab
  labels = [str(i).strip() for i in vocabs.keys()]

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500

  from collections import defaultdict
  wcount = defaultdict(int)
  import sys
  line_id = 0
  with open('All_data.txt') as ofile:
      for line in ofile:
          for w in line.strip().split(" "):
              if w != '':
                  wcount[w] += 1
          if line_id % 100000 == 0:
              sys.stdout.write('\r')
              sys.stdout.write("At line: " + str(line_id))
              sys.stdout.flush()
          line_id += 1

  print("Finished Counting")
  uwords = sorted(wcount, key=wcount.get, reverse=True)

  vectors = [all_model[word] for word in uwords]
  vectors = np.array(vectors)

#  import pdb; pdb.set_trace()
  low_dim_embs = tsne.fit_transform(vectors[:plot_only, :])
  labels_ar = [uwords[i] for i in xrange(plot_only)]
  print ('Plotting ...')
  plot_with_labels(low_dim_embs, labels_ar )

except ImportError:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
