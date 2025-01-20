# Importing libraries
import streamlit as st
import pickle
import torch
import torch.nn.functional as F
from utils import Skipgram, SkipgramNeg, Glove

# Load data and models
Data = pickle.load(open('models\Data.pkl', 'rb'))
corpus = Data['corpus']
vocab = Data['vocab']
word2index = Data['word2index']
voc_size = Data['voc_size']
embed_size = Data['embedding_size']

# Load gensim model
model_save_path = 'models\gensim_model.pkl'
gensim = pickle.load(open(model_save_path, 'rb'))

# Load PyTorch models
skipgram = Skipgram(voc_size, embed_size)
skipgram.load_state_dict(torch.load('models\Skipgram-v1.pt', map_location=torch.device('cpu')))
skipgram.eval()

skipgramNeg = SkipgramNeg(voc_size, embed_size)
skipgramNeg.load_state_dict(torch.load('models\SkipgramNeg-v1.pt', map_location=torch.device('cpu')))
skipgramNeg.eval()

glove = Glove(voc_size, embed_size)
glove.load_state_dict(torch.load('models\GloVe-v1.pt', map_location=torch.device('cpu')))
glove.eval()

# Streamlit UI
st.set_page_config(page_title="Similar Word Search Engine", layout="centered")
st.title("Similar Word Search Engine")
st.write("Please enter your search query below. (Singular words only)")

# Input text box
query = st.text_input("Enter query:")

# Submit button
if st.button("Submit"):
    if query:
        query_words = query.split()
        vectors = []
        results = [[]]  # To store most similar words

        # Gensim Model
        for word in query_words:
            if word in gensim:
                vectors.append(gensim.get_vector(word))
            else:
                vectors.append(gensim.get_vector('unknown'))
        
        # Combine vectors
        result_vector = sum(vectors)
        search = gensim.most_similar(result_vector)

        # Collect results
        for i in range(len(search)):
            results[0].append(search[i][0])

        # For other models
        models = [skipgram, skipgramNeg,glove]
        model_names = ["Skipgram", "SkipgramNeg","GloVe"]
        
        for model_index, model in enumerate(models):
            # Get all word vectors
            all_word_vectors = [model.get_vector(word) for word in vocab]
            all_word_vectors = torch.stack(all_word_vectors)
            
            # Calculate result vector for the input query
            vectors = [
                model.get_vector(word.lower()) if word.lower() in vocab else model.get_vector('<UNK>') 
                for word in query_words
            ]
            result_vector = sum(vectors)
            
            # Calculate cosine similarities
            similarities = F.cosine_similarity(result_vector, all_word_vectors)
            
            # Get indices of the top ten similarities
            top_indices = torch.argsort(similarities, descending=True)[:10]
            top_indices = top_indices.view(-1)  # Ensure it's a 1D tensor
            
            # Fetch the corresponding words from the vocabulary
            top_words = [vocab[index.item()] for index in top_indices[:10]]  # Limit to 10 results
            results.append(top_words)

        # Display results
        st.subheader("Most similar words are:")
        for i, model_name in enumerate(["GloVe (gensim)", *model_names]):  # Include GloVe and the two models
            st.write(f"**{model_name}:**")
            st.table(results[i])
    else:
        st.warning("Please enter a valid query!")
