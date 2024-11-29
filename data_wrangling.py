
#%%
import kagglehub
import pandas as pd


#%%
movies = pd.read_csv('data/movies.csv')
critics = pd.read_csv('data/critic_review.csv')
critics_clean = critics[["rotten_tomatoes_link", "critic_name", "publisher_name", "review_type", "review_score", "review_content"]]


#%%
cleaned = pd.DataFrame()
cleaned = movies[["rotten_tomatoes_link", "movie_title", "critics_consensus", "directors", "tomatometer_rating", "tomatometer_count", "audience_rating", "audience_count"]]

# Assuming 'col2' is the column you want to join on
print(f"Pre merge: {cleaned.shape}")
cleaned = pd.merge(cleaned, critics_clean, on='rotten_tomatoes_link', how='left')
print(f"post merge: {cleaned.shape}")



movie_group = cleaned.groupby("rotten_tomatoes_link")
movie_group.head()


#%%
# %%