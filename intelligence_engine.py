#%%
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import pandas as pd
# .venv/Scripts/activate

#%%
# Analyze please

data_stored = pd.read_csv('output.csv')
average_sentiment = data_stored.groupby('rotten_tomatoes_link').agg({
    'sentiment_batched': 'mean',
    'tomatometer_rating': 'mean'  # Assuming you want to keep the first occurrence of tomatometer_rating
}).reset_index()
average_sentiment['sentiment_norm'] = average_sentiment['sentiment_batched']/5
average_sentiment['tomato_norm'] = average_sentiment['tomatometer_rating']/100
average_sentiment





# %%
# Analyze Critic Reviews
print(f"Before dropna: {average_sentiment.shape}")
cleaned_sentiment = average_sentiment.dropna()
print(f"After dropna: {cleaned_sentiment.shape}")

# Calculate Pearson correlation coefficient
correlation, p_value = pearsonr(cleaned_sentiment['sentiment_norm'], cleaned_sentiment['tomato_norm'])
print(f"Pearson correlation coefficient: {correlation}")
print(f"P-value: {p_value}")

# Visualize the relationship using a scatter plot
plt.figure(figsize=(10, 6))
density = plt.hexbin(cleaned_sentiment['sentiment_norm'], cleaned_sentiment['tomato_norm'], gridsize=30, cmap='Blues')
plt.colorbar(density, label='Density')
plt.title('Scatter plot of sentiment_norm vs. tomato_norm')
plt.xlabel('Normalized Sentiment Score')
plt.ylabel('Normalized Tomatometer Rating')
plt.show()

# %%
