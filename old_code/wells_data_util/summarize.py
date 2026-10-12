import os
import pandas as pd
from pathlib import Path
import json

def get_unique_layers_and_mice(aligned_data_path):
    """
    Extract unique layers and mouse IDs from aligned_data.json.
    
    Parameters:
    -----------
    aligned_data_path : str
        Path to aligned_data.json file
        
    Returns:
    --------
    tuple
        (unique_layers, unique_mice)
    """
    try:
        with open(aligned_data_path, 'r') as f:
            data = json.load(f)
        
        # Extract unique layers from neural metadata
        unique_layers = set()
        for mouse_data in list(data['neural']['metadata'].items()):
            for neuron in mouse_data[1]:
                if 'layer' in neuron:
                    unique_layers.add(neuron['layer'])
        
        # Extract unique mouse IDs from behavior data
        unique_mice = set(data['behavior']['data'].keys())
        
        return sorted(list(unique_layers)), sorted(list(unique_mice))
    except Exception as e:
        print(f"Error reading aligned_data.json: {str(e)}")
        return [], []

def combine_csv_files(root_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store dataframes by filename
    filename_dfs = {}
    
    # Walk through all directories
    for root, dirs, files in os.walk(root_dir):
        # Get the relative path from root directory
        rel_path = os.path.relpath(root, root_dir)
        
        # Split the relative path into components
        path_components = rel_path.split(os.sep)
        
        # Process each CSV file in the current directory
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                
                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)
                    
                    # Add folder structure columns
                    for i, component in enumerate(path_components):
                        if component != '.':
                            df[f'folder_level_{i+1}'] = component
                    
                    # Store the dataframe in the filename_dfs dictionary
                    if file not in filename_dfs:
                        filename_dfs[file] = []
                    filename_dfs[file].append(df)
                    
                    print(f"Processed: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    if not filename_dfs:
        print("No CSV files found!")
        return
    
    # Save individual CSVs for each unique filename
    for filename, dfs in filename_dfs.items():
        if len(dfs) > 0:
            # Remove .csv extension for the output filename
            base_name = os.path.splitext(filename)[0]
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Create a subdirectory for each type of CSV
            csv_output_dir = os.path.join(output_dir, base_name)
            os.makedirs(csv_output_dir, exist_ok=True)
            
            # Save the combined CSV
            output_path = os.path.join(csv_output_dir, f'master_{base_name}.csv')
            combined_df.to_csv(output_path, index=False)
            print(f"\nCreated master CSV for {filename}")
            print(f"Saved to: {output_path}")
            print(f"Total rows: {len(combined_df)}")
            print(f"Total columns: {len(combined_df.columns)}")
            
            # Save a summary of the folder structure
            folder_summary = combined_df[[col for col in combined_df.columns if col.startswith('folder_level_')]].drop_duplicates()
            summary_path = os.path.join(csv_output_dir, f'{base_name}_folder_summary.csv')
            folder_summary.to_csv(summary_path, index=False)
            print(f"Folder structure summary saved to: {summary_path}")

if __name__ == "__main__":
    # Get the current directory
    
    # Get unique layers and mouse IDs
    aligned_data_path = "aligned_data.json"
    unique_layers, unique_mice = get_unique_layers_and_mice(aligned_data_path)
    
    print("\nUnique Layers:")
    for layer in unique_layers:
        print(f"- {layer}")
    
    print("\nUnique Mouse IDs:")
    for mouse_id in unique_mice:
        print(f"- {mouse_id}")
    
    # print(f"\nProcessing CSV files in: {current_dir}")
    # print(f"Saving results to: {output_dir}")
    # combine_csv_files(current_dir, output_dir)
