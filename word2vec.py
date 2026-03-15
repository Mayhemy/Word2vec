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
        print(f"File {self.embeddings_path} not found! Falling back to hardcoded text in DataLoader get_text() function.")
        return "Random text for embeddings this will not converge fast due to it being very short"

def prepare_data(text):
    #remove the unnecessary characters
    for char in ['.', ',', '!', '?', ':', ';']:
        text = text.replace(char,' ')
    words = text.lower().split()

    vocab = list(set(words))
    vocab_size = len(vocab)
    
    lookup_word2idx = {}
    for idx, word in enumerate(vocab):
      lookup_word2idx[word] = idx
  
    corpus = []
    for word in words:
      corpus.append(lookup_word2idx[word])
      
    return vocab, vocab_size, lookup_word2idx, corpus, words
      
      
class Word2Vec_SkipGram_Naive():
  def __init__(self, dataloader, embedding_dim = 30, window_size = 2, epochs = 15, lr = 0.05):
    self.model_name = "SkipGram Naive"
    self.dataloader = dataloader
    self.embedding_dim = embedding_dim
    self.window_size = window_size
    self.epochs = epochs
    self.lr = lr
    self.numerical_offset = 1e-10
  
    text = self.dataloader.get_text()
    self.vocab, self.vocab_size, self.lookup_word2idx, self.corpus, _ = prepare_data(text)

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
      
      grad_w1_accumulated = np.zeros(self.embedding_dim)
      grad_w2_accumulated = np.zeros((self.vocab_size, self.embedding_dim))

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
        
        # gradient update
        # dL/dcurr_center_word_weight = error * w2 -> chain rule for products here
        for p in range(self.vocab_size):
          word_error = error_array[p]
          # += because we have to accumulate gradients
          for q in range(self.embedding_dim):
            grad_w1_accumulated[q] += word_error * self.w2[p][q]
            grad_w2_accumulated[p][q] += word_error * curr_center_word_weights[q]

        lower_bound += 1
        
      for p in range(self.vocab_size):
        for q in range(self.embedding_dim):
          self.w2[p][q] -= self.lr * grad_w2_accumulated[p][q]

      for q in range(self.embedding_dim):
        self.w1[lookup_idx][q] -= self.lr * grad_w1_accumulated[q]

    return overall_loss

class Word2Vec_SkipGram_Optimized():
  def __init__(self, dataloader, embedding_dim = 30, window_size = 2, epochs = 100, lr = 0.05, k_negative_samples = 6):
    self.model_name = "SkipGram Optimized"
    self.dataloader = dataloader
    self.embedding_dim = embedding_dim
    self.window_size = window_size
    self.epochs = epochs
    self.lr = lr
    self.numerical_offset = 1e-10
    self.k_negative_samples = k_negative_samples
    
    text = self.dataloader.get_text()
    self.vocab, self.vocab_size, self.lookup_word2idx, self.corpus, words = prepare_data(text)
    
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    temp_unigram_counts = []
    for word in self.vocab:
        temp_unigram_counts.append(word_counts[word])
     
    #the original code and the Distributed Representations of Words and Phrases and their Compositionality
    unigram_counts = np.array(temp_unigram_counts)
    powered_unigram_counts = unigram_counts ** 0.75
    self.custom_paper_distribution = powered_unigram_counts / np.sum(powered_unigram_counts)
    
    limit = math.sqrt(6/(self.vocab_size + embedding_dim))
    self.w1 = np.random.uniform(-limit, limit, size=(self.vocab_size, self.embedding_dim))
    self.w2 = np.random.uniform(-limit, limit, size=(self.vocab_size, self.embedding_dim))


  def sigmoid(self, x):
    return 1/(1 + np.exp(-np.clip(x, -10, 10)))
    
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
    
  def train(self):
    overall_loss = 0.0
    for i, lookup_idx in enumerate(self.corpus):
      lower_bound = max(i - self.window_size, 0)
      upper_bound = min(i + self.window_size + 1, len(self.corpus))
      curr_center_word_weights = self.w1[lookup_idx]
      
      grad_w1_accumulated = np.zeros(self.embedding_dim)
      grad_w2_accumulated = np.zeros((self.vocab_size, self.embedding_dim))

      while lower_bound < upper_bound:
        if lower_bound == i:
          lower_bound += 1
          continue
          
        positive_idx = self.corpus[lower_bound]
        
        negative_indices = np.random.choice(self.vocab_size, size = self.k_negative_samples, p = self.custom_paper_distribution)


        # not needed anymore since I hardcoded the 1 and 0 while calculating gradients
        # labels_positive = np.ones(1)
        # labels_negatives = np.zeros(self.num_negative_samples)
        
        # forward
        # we can group positive and negative weights into one array and labels into another
        # so just concatenating the arrays since we will anyways be taking the dot product of the target_weights and center word weights
        # however due almost no overhead, we will keep them separate for simpler code understanding
       
        target_weights_positive = self.w2[positive_idx]
        target_weights_negatives = self.w2[negative_indices]
        
        dot_product_positive = np.dot(target_weights_positive, curr_center_word_weights)
        dot_products_negatives = np.dot(target_weights_negatives, curr_center_word_weights)
        
        prediction_positive = self.sigmoid(dot_product_positive)
        prediction_negatives = self.sigmoid(dot_products_negatives)

        # loss
        positive_loss = -np.log(prediction_positive + self.numerical_offset)
        negative_loss = -np.sum(np.log( 1 - prediction_negatives + self.numerical_offset))
        overall_loss += positive_loss + negative_loss

        # since sigmoid, that is used in negative sampling, is just a special case(binary) softmax
        # the gradient for all of the exponential family functions as previously stated in naive implementation comment is always prediction - actual_val
        # that is due to some neat maths
        
        # simple chain rule just like in andrej karpathy's micrograd :)
        # gradient for dL/dz
        error_positive = prediction_positive - 1.0 # label is 1 for positive
        error_negatives = prediction_negatives - 0 # label 0 for negative
        
        # gradient update
        # doing the same as before: dL/dcurr_center_word_weight = error * w2 -> chain rule for products here
        grad_w1_accumulated += error_positive * target_weights_positive
        grad_w1_accumulated += np.dot(error_negatives , target_weights_negatives)
        
        # dL/target_weights = error * w1
        grad_w2_positive = error_positive * curr_center_word_weights
        grad_w2_negatives = np.outer(error_negatives, curr_center_word_weights)
        
        # first update the positive, this can all of these updates can be done on one array
        self.w2[positive_idx] -= self.lr * grad_w2_positive
        
        self.w2[negative_indices] -= self.lr * grad_w2_negatives
        
        lower_bound += 1
        
      self.w1[lookup_idx] -= self.lr * grad_w1_accumulated

    return overall_loss

def test_model(model, test_words):
    for word in test_words:
        print(f"Finding words near '{word}'")
        similarities = model.get_similar_words(word, top_k = 5)
        
        if isinstance(similarities, str):
            print(similarities)
        else:
            for similar_word, score in similarities:
                print(f"Similar word: {similar_word}, score: {score}")
                
def train_eval(model, test_words):
    print(f"Testing model {model.model_name}")
    start_time = time.time()
    for epoch in range(model.epochs):
      loss = model.train()
      #if epoch % 10 == 0:
      print(f"Epoch {epoch+1}/{model.epochs}, loss: {loss}")
      
    train_time = time.time() - start_time
    print(f"Training complete in {train_time} seconds!")
    print("Testing...")
    test_model(model, test_words)
    
    return train_time, loss
    
    
if __name__ == "__main__":
  argument_parser = argparse.ArgumentParser()
  argument_parser.add_argument("-m", "--mode", required=True, help="Choose a mode to run this program in - options: 'naive', 'optimized', 'compare'. Compare compares execution times of both approaches.")
  args = argument_parser.parse_args()
  
  dataloader = DataLoader()
  test_words = ["risc-v", "hardware", "systems"]

  if args.mode == 'naive':
    print("Naive model chosen")
    model_naive = Word2Vec_SkipGram_Naive(dataloader)
    print(f"Naive model hyperparams \nepochs: {model_naive.epochs}, embedding_dim = {model_naive.embedding_dim}, window_size = {model_naive.window_size}, learning_rate = {model_naive.lr}.")
    train_eval(model_naive, test_words)
  elif args.mode == 'optimized':
    print("Optimized model chosen")
    model_optimized = Word2Vec_SkipGram_Optimized(dataloader)
    print(f"Optimized model hyperparams \nepochs: {model_optimized.epochs}, embedding_dim = {model_optimized.embedding_dim}, window_size = {model_optimized.window_size}, learning_rate = {model_optimized.lr}, negative_samples = {model_optimized.k_negative_samples}.")
    train_eval(model_optimized, test_words)
  elif args.mode == 'compare':
    print("Comparison chosen")
    model_naive = Word2Vec_SkipGram_Naive(dataloader)
    naive_time, naive_loss = train_eval(model_naive, test_words)
    
    print("\n\n\n")

    model_optimized = Word2Vec_SkipGram_Optimized(dataloader)
    optimized_time, optimized_loss = train_eval(model_optimized, test_words)


    print("\n\n\n")
    print(30*"=")
    print(f"Optimized model hyperparams \nepochs: {model_optimized.epochs}, embedding_dim = {model_optimized.embedding_dim}, window_size = {model_optimized.window_size}, learning_rate = {model_optimized.lr}, negative_samples = {model_optimized.k_negative_samples}.")
    print(f"Naive model hyperparams \nepochs: {model_naive.epochs}, embedding_dim = {model_naive.embedding_dim}, window_size = {model_naive.window_size}, learning_rate = {model_naive.lr}.")
    print(f"Optimized model is {naive_time - optimized_time} seconds faster than Naive model.")
    print(f"Optimized loss : {optimized_loss}, Naive loss: {naive_loss}.")
  else:
    print("Wrong mode argument!")