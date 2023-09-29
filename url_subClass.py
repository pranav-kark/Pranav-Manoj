import pandas as pd
import tldextract

# Define a function to categorize websites based on their TLDs
def categorize_website(url):
    extracted = tldextract.extract(url)
    if extracted.suffix == 'com':
        return 'Commercial'
    elif extracted.suffix == 'org':
        return 'Organizational'
    elif extracted.suffix == 'edu':
        return 'Educational'
    elif extracted.suffix == 'gov':
        return 'Governmental'
    else:
        return 'Other'

# Define a function to process the URL and classify websites
def classify_websites(url_path, output_path):
    # Read the input CSV file
    df = pd.read_csv(url_path, encoding='utf-8')
    
    # Apply the categorize_website function to the 'url' column and create a new column 'website_type'
    df['website_type'] = df['URL'].astype(str).apply(categorize_website)
    
    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_path, index=False)

# Example usage
url_path = 'Users/KnowledgeArk.AI/Kark/Pranav/url_subClass.py'
output_path = 'output_file_OUTPUT.csv'
classify_websites(url_path, output_path)
