import subprocess
import sys

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