# Collaborative Filtering
#### **Datasets:** 
- MovieLens 
- Netflix Prize Dataset

#### **Preprocessing Steps:**
1. **Load the Dataset:** Use pandas or PySpark, the choice is based on the dataset size (Volume).
2. **Create User-Item Interaction Matrix:** Pivot the data into a sparse matrix format.
3. **Handle Missing Data:** Use mean imputation or leave it sparse for implicit feedback models.

#### **Code Example:**
**Matrix Factorization with Surprise (SVD)**

```python
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split, accuracy

# Load MovieLens Dataset
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.2)

# Train SVD model
model = SVD()
model.fit(trainset)

# Predictions and Evaluation
predictions = model.test(testset)
accuracy.rmse(predictions)
```

# Content-Based Recommendation
#### **Datasets:**
- IMDb Dataset
- Goodreads Dataset

#### **Preprocessing Steps:**
1. **Extract Metadata:** Parse genres, authors, directors, etc.
2. **Feature Engineering:** Create TF-IDF or one-hot encoding for categorical features.
3. **Build User Profiles:** Aggregate metadata of liked items for personalized recommendations.

#### **Code Example:**
**Content-Based Filtering with TF-IDF**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

# Load IMDb Dataset
movies = pd.read_csv('movies_metadata.csv')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'].fillna(''))

# Compute Similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Recommend Movies
def recommend(movie_title, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return [movies['title'][i[0]] for i in sim_scores[1:6]]

print(recommend('The Godfather'))
```


# Hybrid Models
#### **Datasets:**
- Yelp Dataset
- Amazon Product Data

#### **Preprocessing Steps:**
1. **Collaborative Filtering:** Create a user-item matrix.
2. **Content-Based Filtering:** Engineer features from metadata.
3. **Combine Scores:** Weight or stack CF and CB models for final recommendations.

#### **Code Example:**
**Weighted Hybrid Model**

```python
import numpy as np

# Collaborative Filtering Scores
cf_scores = np.array([4.5, 3.8, 5.0, 4.2])

# Content-Based Filtering Scores
cb_scores = np.array([3.0, 4.2, 4.8, 4.0])

# Weighted Hybrid
weights = [0.6, 0.4]
final_scores = weights[0] * cf_scores + weights[1] * cb_scores
print("Final Recommendations:", final_scores.argsort()[::-1])
```

# Implicit Feedback Models
#### **Datasets:**
- Instacart Market Basket
- E-Commerce Behavior Dataset

#### **Preprocessing Steps:**
1. **Binary Relevance Matrix:** Convert interactions (e.g., purchases) into binary values.
2. **Train Implicit Models:** Use models like Alternating Least Squares (ALS).

#### **Code Example:**
**Implicit Collaborative Filtering (ALS)**

```python
from implicit.als import AlternatingLeastSquares
import scipy.sparse as sparse

# Create Sparse Matrix
user_item_matrix = sparse.csr_matrix((data['interaction'], 
                                      (data['user_id'], data['item_id'])))

# Train ALS Model
model = AlternatingLeastSquares(factors=50, regularization=0.1)
model.fit(user_item_matrix)

# Recommend for User
user_id = 0
recommendations = model.recommend(user_id, user_item_matrix[user_id])
print("Recommendations:", recommendations)
```


# Sequential/Session-Based Models
#### **Datasets:**
- Spotify Million Playlist
- E-Commerce Clickstream Data

#### **Preprocessing Steps:**
1. **Sort by Timestamps:** Order interactions sequentially.
2. **Prepare Input Sequences:** Use session-level splitting.
3. **Train Sequential Models:** Apply GRU4Rec or Transformers.

#### **Code Example:**
**GRU4Rec for Sequential Recommendations**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense

# Dummy Sequential Data
X = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
y = [4, 5, 6]

# Build GRU Model
model = Sequential([
    Embedding(input_dim=10, output_dim=50, input_length=3),
    GRU(64, return_sequences=False),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X, y, epochs=10)
```


# Graph-Based Recommendation
#### **Datasets:**
- Last.fm Dataset
- Amazon Product Data

#### **Preprocessing Steps:**
1. **Build Graph:** Create user-item bipartite graphs.
2. **Node Embeddings:** Use GCN or PinSage to compute representations.
3. **Train Graph Models:** Predict links for recommendations.

#### **Code Example:**
**Graph-Based Recommendations (NetworkX + Node2Vec)**

```python
import networkx as nx
from node2vec import Node2Vec

# Create Graph
G = nx.Graph()
G.add_edges_from([(0, 'A'), (0, 'B'), (1, 'B'), (1, 'C')])

# Train Node2Vec
node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=100)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Recommend Items
user_embedding = model.wv['0']  # User ID 0
item_similarities = {item: model.wv.similarity('0', item) for item in ['A', 'B', 'C']}
print("Recommendations:", sorted(item_similarities, key=item_similarities.get, reverse=True))
```

# Tools & Libraries
- **General:** Pandas, NumPy
- **Collaborative Filtering:** Surprise, Implicit
- **Content-Based:** Scikit-learn, NLTK
- **Deep Learning:** TensorFlow, PyTorch
- **Graph-Based:** NetworkX, StellarGraph, PyTorch Geometric
