import os
import pandas as pd
from tqdm import tqdm
import re

def parse_metadata(file_path: str) -> dict:
    """
    Parse the metadata from a given file and return it as a dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
        # Remove the enclosing angle brackets
        content = content.strip('<> ')

        # Split the content based on spaces outside of quotes
        attributes = re.findall(r'(\S+)=["\']([^"\']*)["\']', content)
        
        # Convert list of tuples into dictionary
        return dict(attributes)

def process_directory(directory: str) -> None:
    """
    Process all metadata files in the given directory and
    save the data to a CSV file.
    """

    data = []
    
    # Walk through all files in the directory   
    for filename in tqdm(os.listdir(directory)):

        if filename.endswith(".metadata"):
            file_path = os.path.join(directory, filename)

            try:
                attributes = parse_metadata(file_path)
                data.append(attributes)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Save DataFrame to CSV
    df.to_csv(
        "../data/outputs/rsc_metadata_full.csv", 
        index=False , 
        encoding='utf-8'
    )

# Use the function on your specific directory
directory_path = "../data/inputs/metadata_raw/" 
process_directory(directory_path)
