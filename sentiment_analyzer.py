#%%
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F



# Load the pretrained model and tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to perform sentiment analysis
def sentiment_analysis(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction2 = F.softmax(logits, dim=-1)
    scores = torch.arange(5).float() @ prediction2.T
    return scores

# Test usage
texts = ["I love you so much.", "This is the worst experience I've ever had.", "This was neutral.", "I want to have your babies"]
prediction1 = sentiment_analysis(texts)
print(f"test:   {prediction1}")  # Output: tensor([1, 0]) where 1 is positive and 0 is negative




#%%
#Analyze Critic Reviews
from data_wrangling import cleaned
from data_wrangling import critics_clean

print(f"before dropna: {cleaned.shape}")
cleanedna= cleaned.dropna(subset=['review_score', 'review_content'])
print(f"after dropna: {cleanedna.shape}")
test5 = cleanedna.head(5000)


#%% SENTIMENT - ROUGLY 50 ROWS / SECOND
test5['sentiment'] = test5['review_content'].apply(lambda x: sentiment_analysis([x])[0].item())













# # %% Batch It!

# # Function to perform sentiment analysis on batches
# def sentiment_analysis_batch(texts):
#     inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits = outputs.logits
#     predictions = F.softmax(logits, dim=-1)
#     scores = torch.arange(1, 6).float().to("cuda") @ predictions.T  # Weighted sum for star ratings
#     return scores.cpu().tolist()

# # Apply batch processing to the DataFrame
# batch_size = 64  # Adjust batch size based on your system's memory
# results = []
# for i in range(0, len(test5), batch_size):
#     batch_texts = test5['review_content'].iloc[i:i+batch_size].tolist()
#     results.extend(sentiment_analysis_batch(batch_texts))

# # Add the results as a new column
# test5['sentiment_batched'] = results

# # %%
