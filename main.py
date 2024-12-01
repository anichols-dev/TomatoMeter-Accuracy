#%%
import subprocess
import sys
import pandas as pd
from scipy.stats import pearsonr
import kagglehub
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from data_wrangling import cleaned


#%%
# List of scripts to run
scripts = ['data_wrangling.py', 'sentiment_analyzer.py', 'intelligence_engine.py']

def run_script(script):
    try:
        # Run the script using the python interpreter
        subprocess.run([sys.executable, script], check=True)
        print(f"Successfully ran {script}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to run {script}: {e}")

if __name__ == "__main__":
    for script in scripts:
        run_script(script)

# #%%
# from intelligence_engine import analyze_sentiments, visualize_results

# # Analyze sentiments from the CSV file
# cleaned_data, inlier_data = analyze_sentiments('output.csv')

# # Visualize the results
# visualize_results(cleaned_data, inlier_data)