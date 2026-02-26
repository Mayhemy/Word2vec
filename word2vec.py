import numpy as np
import multiprocessing as mp
import math
import time
import argparse
from pathlib import Path
#import matplotlib.pyplot as plt

class DataLoader:
  #hardcoded dataloader
  def __init__(self, embeddings_file_path = './input.txt'):
    #Tiny Shakespeare from Andrej Karpathy
    self.embeddings_path = Path(embeddings_file_path)
    
  def get_text(self):
    try:
        return self.embeddings_path.read_text(encoding = 'utf-8')
    except FileNotFoundError:
        print(f"File {embeddings_file_path} not found! Falling back to hardcoded text in DataLoader get_text() function.")
        return "Random text for embeddings this will not converge fast due to it being very short"

class Word2Vec_SkipGram_Naive():
  def __init__(self, dataloader, embedding_dim = 30, window_size = 2, epochs = 15, lr = 0.05):
    self.dataloader = dataloader
    self.embedding_dim = embedding_dim
    self.window_size = window_size
    self.epochs = epochs
    self.lr = lr
    self.numerical_offset = 1e-10
  
    text = self.dataloader.get_text()
    #remove the unnecessary characters
    for char in ['.', ',', '!', '?', ':', ';']:
        text = text.replace(char,' ')
    words = text.lower().split()

    self.vocab = list(set(words))
    self.vocab_size = len(self.vocab)
    
    self.lookup_word2idx = {}
    for idx, word in enumerate(self.vocab):
      self.lookup_word2idx[word] = idx
  
    self.corpus = []
    for word in words:
      self.corpus.append(self.lookup_word2idx[word])

    # init both weight arrays, center(w1) and context(w2) and since we will be using softmax, we do xavier init where var = 2/(n_in + n_out) with normal distribution
    # but we use the other form with uniform distribution due to numerical instability while keeping the same variance => sqrt(6/ (n_in + n_out)), 
    # since probability must be 1 in the range from (-limit, + limit) the height is 1/(high-low) which is equal to 1/2*limit
    # var = E[X^2] - E[X]^2 and mean (E[X]) is 0, because the range is (-limit, +limit) and integral x * 1/2*limit from -limit to limit is 0 of course,
    # we calculate only var = E[X^2] which is integral of x^2 * 1/2*limit and get var = limit^2/3, so this tells us that we have to multiply the uniform distribution limits by sqrt(3) to stay in the same range as normal dist
    limit = math.sqrt(6/(self.vocab_size + embedding_dim))
    self.w1 = np.random.uniform(-limit, limit, size=(self.vocab_size, self.embedding_dim))
    self.w2 = np.random.uniform(-limit, limit, size=(self.vocab_size, self.embedding_dim))

  def softmax(self, dot_products):
    # log-sum-exp trick
    adjusted_dot_products = np.exp(dot_products - np.max(dot_products))
    return adjusted_dot_products/np.sum(adjusted_dot_products)

  def get_similar_words(self, searched_word, top_k = 5):
    if searched_word not in self.lookup_word2idx:
      return "Word not found"

    searched_word_idx = self.lookup_word2idx[searched_word]
    searched_word_weights = self.w1[searched_word_idx]

    searched_word_norm = np.linalg.norm(searched_word_weights)

    list_of_similarities = []
    for i, word in enumerate(self.vocab):
      if word == searched_word:
        continue
      
      word_weights = self.w1[self.lookup_word2idx[word]]
      word_norm = np.linalg.norm(word_weights)
      # cosine similarity
      dot_product = np.dot(searched_word_weights, word_weights)
      norm_product = searched_word_norm * word_norm
      cos_similarity = dot_product / norm_product

      list_of_similarities.append((word, cos_similarity))

    list_of_similarities.sort(key = lambda x: x[1])
    list_of_similarities.reverse()
    return list_of_similarities[:top_k]
  
  # would have to do PCA for embedding_dim > 2 which wouldn't be that useful I guess, so sticking to similar words test
  def draw_plot(self):
    plt.figure()

  def train(self):
    overall_loss = 0.0
    for i, lookup_idx in enumerate(self.corpus):
      lower_bound = max(i - self.window_size, 0)
      upper_bound = min(i + self.window_size + 1, len(self.corpus))
      curr_center_word_weights = self.w1[lookup_idx]

      while lower_bound < upper_bound:
        if lower_bound == i:
          lower_bound += 1
          continue
        context_idx = self.corpus[lower_bound]

        # forward
        dot_products = np.dot(self.w2, curr_center_word_weights)

        # loss
        y_hat = self.softmax(dot_products)
        loss = -np.log(y_hat[context_idx] + self.numerical_offset)
        overall_loss += loss

        # simple chain rule just like in andrej karpathy's micrograd :)
        # gradient for dL/dz
        y_hat[context_idx] -= 1 # for any exponential family model gradients are calculated as y_hat - y, and here since y is one_hot we can just do this.
        error_array = y_hat
        
        #dL/dcurr_center_word_weight = error * w2 -> chain rule for products here
        grad_curr_center_word_weights = np.zeros(self.embedding_dim)
        grad_w2 = np.zeros((self.vocab_size, self.embedding_dim))

        for p in range(self.vocab_size):
          word_error = error_array[p]
          # += because we have to accumulate gradients
          for q in range(self.embedding_dim):
            grad_curr_center_word_weights[q] += word_error * self.w2[p][q]
          
          # right now += is not needed here because each context word affects only one thing and that is center word, this will change if we introduce batching so we will keep += since it doesnt impact anything right now
          for q in range(self.embedding_dim):
            grad_w2[p][q] += word_error * curr_center_word_weights[q]
        
        for p in range(self.vocab_size):
          for q in range(self.embedding_dim):
            self.w2[p][q] -= self.lr * grad_w2[p][q]

        for q in range(self.embedding_dim):
          self.w1[lookup_idx][q] -= self.lr * grad_curr_center_word_weights[q]

        lower_bound += 1

    return overall_loss

#TODO
class Word2Vec_SkipGram_Optimized():
  def __init__(self, dataloader, embedding_dim = 2, window_size = 2, lr = 0.03):
    self.dataloader = dataloader
    self.embedding_dim = embedding_dim
    self.window_size = window_size
    self.lr = lr
    self.numerical_offset = 1e-10

  def train(self):
    return 0

if __name__ == "__main__":
  argument_parser = argparse.ArgumentParser()
  argument_parser.add_argument("-m", "--mode", required=True, help="Choose a mode to run this program in - options: 'naive', 'optimized', 'compare'. Compare compares execution times of both approaches.")
  args = argument_parser.parse_args()
  
  dataloader = DataLoader()

  if args.mode == 'naive':
    print("Naive model chosen")
    model_naive = Word2Vec_SkipGram_Naive(dataloader)
    for epoch in range(model_naive.epochs):
      loss = model_naive.train()
      #if epoch % 10 == 0:
      print(f"Epoch {epoch+1}, loss: {loss}")
    print("Training complete!")
    print("Testing...")
    
    test_words = ['risc-v', 'hardware', 'systems']
    
    for word in test_words:
        print(f"Finding words near '{word}'")
        similarities = model_naive.get_similar_words(word, top_k = 5)
        
        if isinstance(similarities, str):
            print(similarities)
        else:
            for similar_word, score in similarities:
                print(f"Similar word: {similar_word}, score: {score}")
  # isnt implemented yet
  elif args.mode == 'optimized':
    print("Optimized model chosen")
    model_optimized = Word2Vec_SkipGram_Optimized(dataloader)
    loss = model_optimized.train()
    print(f"Optimized Loss is {loss}")
  # isnt implemented yet
  elif args.mode == 'compare':
    print("Optimized model chosen")
    model_naive = Word2Vec_SkipGram_Naive(dataloader)
    start_time_naive = time.time()
    loss_naive = model_naive.train()
    naive_time = time.time() - start_time_naive
    print(f"Naive Loss is {loss_naive}")

    model_optimized = Word2Vec_SkipGram_Optimized(dataloader)
    start_time_optimized = time.time()
    loss_optimized = model_optimized.train()
    optimized_time = time.time() - start_time_optimized
    print(f"Optimized Loss is {loss_optimized}")

    if optimized_time > 0:
      print(f"Optimized code is {naive_time/optimized_time}x faster")
    else:
      print(f"Optimized code is {naive_time - optimized_time} seconds faster")
  else:
    print("Wrong mode argument!")