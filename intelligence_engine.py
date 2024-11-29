#%%
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sentiment_analyzer import test5


#%%
# Analyze please
average_sentiment = test5.groupby('rotten_tomatoes_link').agg({
    'sentiment': 'mean',
    'tomatometer_rating': 'mean'  # Assuming you want to keep the first occurrence of tomatometer_rating
}).reset_index()
average_sentiment['sentiment_norm'] = average_sentiment['sentiment']/5
average_sentiment['tomato_norm'] = average_sentiment['tomatometer_rating']/100
average_sentiment





# %%

# Calculate Pearson correlation coefficient
correlation, p_value = pearsonr(average_sentiment['sentiment_norm'], average_sentiment['tomato_norm'])
print(f"Pearson correlation coefficient: {correlation}")
print(f"P-value: {p_value}")

# Visualize the relationship using a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sentiment_norm', y='tomato_norm', data=average_sentiment)
plt.title('Scatter plot of sentiment_norm vs. tomato_norm')
plt.xlabel('Normalized Sentiment Score')
plt.ylabel('Normalized Tomatometer Rating')
plt.show()

# %%
