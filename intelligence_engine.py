#%%
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from data_wrangling import cleaned


#%%

def analyze_sentiments(filepath):
    # Read data from CSV
    data_stored = pd.read_csv(filepath)
    
    # Group by 'rotten_tomatoes_link' and calculate mean values
    average_sentiment = data_stored.groupby('rotten_tomatoes_link').agg({
        'sentiment_batched': 'mean',
        'tomatometer_rating': 'mean'
    }).reset_index()
    
    # Normalize sentiment and tomatometer ratings
    average_sentiment['sentiment_norm'] = average_sentiment['sentiment_batched'] / 5
    average_sentiment['tomato_norm'] = average_sentiment['tomatometer_rating'] / 100
    
    # Drop missing values
    print(f"Before dropna: {average_sentiment.shape}")
    cleaned_sentiment = average_sentiment.dropna()
    print(f"After dropna: {cleaned_sentiment.shape}")

    # Calculate Pearson correlation coefficient
    correlation, p_value = pearsonr(cleaned_sentiment['sentiment_norm'], cleaned_sentiment['tomato_norm'])
    print(f"Pearson correlation coefficient: {correlation}")
    print(f"P-value: {p_value}")

    # Filter inliers for further analysis
    print(f"Before outliers: {cleaned_sentiment.shape}")
    inliers = cleaned_sentiment[(cleaned_sentiment['tomato_norm'] < 1) & (cleaned_sentiment['tomato_norm'] > 0)]
    print(f"After outliers: {inliers.shape}")

    # Recalculate Pearson correlation coefficient for inliers
    correlation_inliers, _ = pearsonr(inliers['sentiment_norm'], inliers['tomato_norm'])
    print(f"Pearson correlation coefficient for inliers: {correlation_inliers}")

    return cleaned_sentiment, inliers

cleaned_sentiment, inliers = analyze_sentiments('output.csv')
#%%


def visualize_results(df_sentiment, inliers):

    # Drop missing values
    print(f"Before dropna: {df_sentiment.shape}")
    cleaned_sentiment = df_sentiment.dropna()
    print(f"After dropna: {cleaned_sentiment.shape}")
    # Hexbin Plot for cleaned sentiment
    plt.figure(figsize=(10, 6))
    density = plt.hexbin(cleaned_sentiment['sentiment_norm'], cleaned_sentiment['tomato_norm'], gridsize=30, cmap='Blues')
    plt.colorbar(density, label='Density')
    plt.title('Hexbin plot of sentiment_norm vs. tomato_norm (All Data)')
    plt.xlabel('Normalized Sentiment Score')
    plt.ylabel('Normalized Tomatometer Rating')
    plt.show()

    # Joint Plot for cleaned sentiment
    sns.jointplot(x='sentiment_norm', y='tomato_norm', data=cleaned_sentiment, kind='scatter', alpha=0.5)
    plt.suptitle('Joint plot of sentiment_norm vs. tomato_norm (All Data)', y=1.02)
    plt.show()
    
    # Hexbin Plot for inliers
    plt.figure(figsize=(10, 6))
    density = plt.hexbin(inliers['sentiment_norm'], inliers['tomato_norm'], gridsize=30, cmap='Blues')
    plt.colorbar(density, label='Density')
    plt.title('Hexbin plot of sentiment_norm vs. tomato_norm (Inliers)')
    plt.xlabel('Normalized Sentiment Score')
    plt.ylabel('Normalized Tomatometer Rating')
    plt.show()

    # Joint Plot for inliers
    sns.jointplot(x='sentiment_norm', y='tomato_norm', data=inliers, kind='scatter', alpha=0.5)
    plt.suptitle('Joint plot of sentiment_norm vs. tomato_norm (Inliers)', y=1.02)
    plt.show()

#%%

top_movies = cleaned_sentiment[cleaned_sentiment['sentiment_norm'] > cleaned_sentiment['sentiment_norm'].quantile(0.90)]
top_min = top_movies['sentiment_norm'].min()
bad_movies = cleaned_sentiment[cleaned_sentiment['sentiment_norm'] < cleaned_sentiment['sentiment_norm'].quantile(0.10)]
bad_max = bad_movies['sentiment_norm'].max()

print(f'Count top movies: {top_movies.shape[0]} at {top_min:.2f} and count bad movies: {bad_movies.shape[0]} at {bad_max:.2f}')


# %%
