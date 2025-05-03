import json

def analyze_json_structure(data, prefix=""):
    if isinstance(data, dict):
        print(f"{prefix}Object with keys: {list(data.keys())}")
        for key in data:
            if isinstance(data[key], (dict, list)):
                print(f"{prefix}Key: {key}")
                analyze_json_structure(data[key], prefix + "  ")
    elif isinstance(data, list):
        if len(data) > 0:
            print(f"{prefix}List of length {len(data)}")
            if isinstance(data[0], (dict, list)):
                analyze_json_structure(data[0], prefix + "  ")

def main():
    # Analyze reaches data
    print("\nAnalyzing df_reaches.json structure:")
    with open('data/wells_data/df_reaches.json', 'r') as f:
        reach_data = json.load(f)
    analyze_json_structure(reach_data)

    # Analyze neural data
    print("\nAnalyzing df_neural.json structure:")
    with open('data/wells_data/df_neural.json', 'r') as f:
        neural_data = json.load(f)
    analyze_json_structure(neural_data)

if __name__ == "__main__":
    main() 