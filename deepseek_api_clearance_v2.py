import pandas as pd
import requests
import time
import json
import re
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CSV_PATH = "Mock_Input_Features_Sample.csv"
API_KEY = "sk-455bbebad6f9481eafbf2d6cdda52f08"  # Replace with your actual API key
OUTPUT_PATH = "enriched_feature_review_deepseek_v2.xlsx"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
REQUEST_DELAY = 1  # seconds between requests

# Expected field structure for validation
EXPECTED_FIELDS = [
    "Is Derived?",
    "Is Derived? Reason",
    "Used in Modeling?",
    "Used in Modeling? Reason",
    "Transformation Needed?",
    "Transformation Needed? Reason",
    "Time-Dependent?",
    "Time-Dependent? Reason",
    "Bucketed Version Exists?",
    "Bucketed Version Exists? Reason",
    "Interactional Potential?",
    "Interactional Potential? Reason"
]

def generate_prompt(row: pd.Series) -> str:
    """Generate a structured prompt for feature evaluation."""
    return f"""You are evaluating a single feature from a dataset used to predict university student dropout.

Feature Metadata:
- COLUMN NAME: {row['Column Name']}
- DATA TYPE: {row['Data Type']}
- UNIQUE VALUES COUNT: {row['Unique Values Count']}
- % MISSING: {row['% Missing']}
- CARDINALITY: {row['Cardinality']}
- RELATED FIELD: {row['Related Field']}
- POSSIBLE MEANING: {row['Possible Meaning']}
- NOTES: {row['Notes']}

CRITICAL: You must provide exactly 12 lines in the following format. Each line must contain exactly one colon (:) separating the field name from the value.

Required format (copy exactly, replace [...] with your answers):

Is Derived?: [Yes/No]
Is Derived? Reason: [Brief specific explanation for this feature]
Used in Modeling?: [Yes/No/Dropped]
Used in Modeling? Reason: [Brief specific explanation for this feature]
Transformation Needed?: [Yes/No]
Transformation Needed? Reason: [Brief specific explanation for this feature]
Time-Dependent?: [Yes/No]
Time-Dependent? Reason: [Brief specific explanation for this feature]
Bucketed Version Exists?: [Yes/No]
Bucketed Version Exists? Reason: [Brief specific explanation for this feature]
Interactional Potential?: [Yes/No]
Interactional Potential? Reason: [Brief specific explanation for this feature]

Guidelines:
- Answer exactly Yes/No (or Yes/No/Dropped for modeling question)
- Keep reasons under 20 words and specific to this feature
- No markdown, bullets, or extra formatting
- No additional commentary or explanations
- Must be exactly 12 lines total"""

def ask_deepseek(prompt: str, max_retries: int = MAX_RETRIES) -> Optional[str]:
    """Make API request to DeepSeek with retry logic."""
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,  # Lower temperature for more consistent formatting
        "max_tokens": 500,   # Limit response length
        "top_p": 0.9
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if 'choices' not in result or not result['choices']:
                logger.warning(f"Empty response on attempt {attempt + 1}")
                continue
                
            content = result['choices'][0]['message']['content'].strip()
            if content:
                return content
            else:
                logger.warning(f"Empty content on attempt {attempt + 1}")
                
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error on attempt {attempt + 1}: {e}")
        except (KeyError, json.JSONDecodeError) as e:
            logger.warning(f"Response parsing error on attempt {attempt + 1}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
        
        if attempt < max_retries - 1:
            time.sleep(RETRY_DELAY)
    
    logger.error(f"Failed to get response after {max_retries} attempts")
    return None

def parse_response(response: str, row_index: int, column_name: str) -> Dict[str, str]:
    """Parse LLM response into structured dictionary with validation."""
    if not response:
        logger.warning(f"Empty response for row {row_index} ({column_name})")
        return create_empty_entry()
    
    # Split into lines and clean
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    
    # Filter lines that contain exactly one colon
    valid_lines = [line for line in lines if line.count(':') == 1]
    
    if len(valid_lines) != 12:
        logger.warning(f"Row {row_index} ({column_name}): Expected 12 lines, got {len(valid_lines)}")
        # Try to salvage what we can
    
    entry = {}
    used_keys = set()
    
    for line in valid_lines:
        try:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Avoid duplicate keys
            if key in used_keys:
                logger.warning(f"Row {row_index}: Duplicate key '{key}' found")
                continue
            used_keys.add(key)
            
            # Validate key is expected
            if key in EXPECTED_FIELDS:
                entry[key] = value
            else:
                logger.warning(f"Row {row_index}: Unexpected key '{key}'")
                
        except ValueError:
            logger.warning(f"Row {row_index}: Could not parse line: {line}")
            continue
    
    # Fill missing fields with placeholder values
    for field in EXPECTED_FIELDS:
        if field not in entry:
            entry[field] = "ERROR: Missing response"
            logger.warning(f"Row {row_index} ({column_name}): Missing field '{field}'")
    
    # Validate answer formats
    validate_answers(entry, row_index, column_name)
    
    return entry

def validate_answers(entry: Dict[str, str], row_index: int, column_name: str) -> None:
    """Validate that answers follow expected format."""
    binary_fields = [
        "Is Derived?", "Transformation Needed?", 
        "Time-Dependent?", "Bucketed Version Exists?", "Interactional Potential?"
    ]
    
    for field in binary_fields:
        if field in entry:
            answer = entry[field].upper()
            if answer not in ['YES', 'NO']:
                logger.warning(f"Row {row_index} ({column_name}): Invalid answer for '{field}': {entry[field]}")
    
    # Special validation for modeling field
    if "Used in Modeling?" in entry:
        answer = entry["Used in Modeling?"].upper()
        if answer not in ['YES', 'NO', 'DROPPED']:
            logger.warning(f"Row {row_index} ({column_name}): Invalid answer for 'Used in Modeling?': {entry['Used in Modeling?']}")

def create_empty_entry() -> Dict[str, str]:
    """Create empty entry for failed responses."""
    return {field: "ERROR: No response" for field in EXPECTED_FIELDS}

def save_checkpoint(df_out: pd.DataFrame, checkpoint_path: str) -> None:
    """Save intermediate results."""
    try:
        df_out.to_excel(checkpoint_path, index=False)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

def main():
    """Main execution function."""
    # Read CSV
    try:
        df = pd.read_csv(CSV_PATH)
        logger.info(f"Loaded {len(df)} rows from {CSV_PATH}")
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return
    
    # Validate required columns
    required_columns = ['Column Name', 'Data Type', 'Unique Values Count', 
                       '% Missing', 'Cardinality', 'Related Field', 
                       'Possible Meaning', 'Notes']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return
    
    responses = []
    parsed_entries = []
    
    # Process each row
    for index, row in df.iterrows():
        column_name = row['Column Name']
        logger.info(f"Processing row {index+1}/{len(df)}: {column_name}")
        
        prompt = generate_prompt(row)
        response = ask_deepseek(prompt)
        responses.append(response if response else "")
        
        parsed_entry = parse_response(response, index, column_name)
        parsed_entries.append(parsed_entry)
        
        # Save checkpoint every 10 rows
        if (index + 1) % 10 == 0:
            temp_df = pd.concat([df.iloc[:index+1].reset_index(drop=True), 
                               pd.DataFrame(parsed_entries)], axis=1)
            save_checkpoint(temp_df, f"checkpoint_{index+1}.xlsx")
        
        time.sleep(REQUEST_DELAY)
    
    # Create final output
    try:
        df_parsed = pd.DataFrame(parsed_entries)
        df_out = pd.concat([df.reset_index(drop=True), df_parsed], axis=1)
        df_out.to_excel(OUTPUT_PATH, index=False)
        logger.info(f"✅ Final output saved to {OUTPUT_PATH}")
        
        # Generate summary report
        generate_summary_report(df_out, len(df))
        
    except Exception as e:
        logger.error(f"Failed to create final output: {e}")

def generate_summary_report(df_out: pd.DataFrame, total_rows: int) -> None:
    """Generate a summary report of the processing results."""
    error_count = 0
    for field in EXPECTED_FIELDS:
        if field in df_out.columns:
            error_count += df_out[field].str.contains("ERROR:", na=False).sum()
    
    logger.info(f"""
    📊 PROCESSING SUMMARY:
    - Total rows processed: {total_rows}
    - Total fields expected: {total_rows * len(EXPECTED_FIELDS)}
    - Fields with errors: {error_count}
    - Success rate: {((total_rows * len(EXPECTED_FIELDS) - error_count) / (total_rows * len(EXPECTED_FIELDS)) * 100):.1f}%
    """)

if __name__ == "__main__":
    main()