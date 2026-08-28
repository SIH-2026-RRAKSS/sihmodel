import os
import sys

try:
    import kagglehub
except ImportError:
    print("Error: 'kagglehub' is not installed.")
    print("Please run: pip install kagglehub pandas networkx")
    sys.exit(1)

def download_ibm_aml_dataset():
    print("="*60)
    print("Downloading IBM AML Dataset (HI-Small_Trans.csv)")
    print("Warning: This dataset is large (~8.35 GB uncompressed).")
    print("="*60)
    
    # Download latest version from Kaggle
    path = kagglehub.dataset_download("ealtman2019/ibm-transactions-for-anti-money-laundering-aml")
    
    csv_file = os.path.join(path, "HI-Small_Trans.csv")
    if os.path.exists(csv_file):
        print("\n[SUCCESS] Dataset downloaded successfully!")
        print(f"Location: {csv_file}")
        print("\nYou can now run the adapter to generate the graphs:")
        print(f"python src/import_ibm_aml.py --csv-path \"{csv_file}\" --output-dir data/graphs_ibm")
    else:
        print(f"\n[WARNING] Downloaded to {path}, but HI-Small_Trans.csv was not found.")

if __name__ == "__main__":
    download_ibm_aml_dataset()
