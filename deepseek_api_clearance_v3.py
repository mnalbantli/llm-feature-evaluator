import pandas as pd
import requests
import time
import json
import re
import os
from typing import Dict, List, Optional
import logging
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CSV_PATH = "Mock_Input_Features_Sample.csv"
OUTPUT_PATH = "enriched_feature_review_deepseek_v2.xlsx"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
REQUEST_DELAY = 1  # seconds between requests

# Context window configuration
CONTEXT_WINDOW_SIZE = 3  # N rows before and after current row
MAX_PROMPT_TOKENS = 4000  # Approximate token limit for context

# Expected field structure for validation
EXPECTED_FIELDS = [
    "Data Source",
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

# Columns to include in context (excluding the fields we're trying to fill)
CONTEXT_COLUMNS = [
    "Field ID", "Column Name", "Related Field", "Data Type", 
    "Unique Values Count", "% Missing", "Sample Values", 
    "Cardinality", "Possible Meaning", "Notes", "Usage Example"
]

def estimate_token_count(text: str) -> int:
    """Rough estimation of token count (approximately 4 characters per token)."""
    return len(text) // 4

def create_context_window(df: pd.DataFrame, current_index: int, window_size: int = CONTEXT_WINDOW_SIZE) -> Tuple[List[int], pd.DataFrame]:
    """Create a context window around the current row."""
    total_rows = len(df)
    
    # Calculate window boundaries
    start_idx = max(0, current_index - window_size)
    end_idx = min(total_rows, current_index + window_size + 1)
    
    # Get the windowed dataframe
    context_df = df.iloc[start_idx:end_idx].copy()
    
    # Get relative indices for the window
    window_indices = list(range(start_idx, end_idx))
    
    return window_indices, context_df

def format_context_row(row: pd.Series, is_current: bool = False) -> str:
    """Format a single row for context display."""
    prefix = ">>> [CURRENT ROW TO EVALUATE] >>>" if is_current else "    [CONTEXT ROW]"
    
    row_text = f"{prefix}\n"
    for col in CONTEXT_COLUMNS:
        if col in row.index:
            value = str(row[col]) if pd.notna(row[col]) else "N/A"
            row_text += f"  {col}: {value}\n"
    
    return row_text

def generate_contextual_prompt(df: pd.DataFrame, current_index: int) -> str:
    """Generate a prompt with sliding window context."""
    
    # Get context window
    window_indices, context_df = create_context_window(df, current_index)
    current_row = df.iloc[current_index]
    
    # Find the position of current row in the context window
    current_pos_in_window = current_index - window_indices[0]
    
    # Build context section
    context_section = "DATASET CONTEXT (neighboring features for reference):\n"
    context_section += "=" * 60 + "\n"
    
    for i, (_, row) in enumerate(context_df.iterrows()):
        is_current = (i == current_pos_in_window)
        context_section += format_context_row(row, is_current)
        context_section += "-" * 40 + "\n"
    
    # Check if context is too long and truncate if needed
    if estimate_token_count(context_section) > MAX_PROMPT_TOKENS * 0.7:  # Leave room for instructions
        logger.warning(f"Context too long for row {current_index}, truncating...")
        # Reduce window size dynamically
        reduced_window = max(1, CONTEXT_WINDOW_SIZE // 2)
        window_indices, context_df = create_context_window(df, current_index, reduced_window)
        current_pos_in_window = current_index - window_indices[0]
        
        context_section = "DATASET CONTEXT (neighboring features - truncated):\n"
        context_section += "=" * 60 + "\n"
        
        for i, (_, row) in enumerate(context_df.iterrows()):
            is_current = (i == current_pos_in_window)
            context_section += format_context_row(row, is_current)
            context_section += "-" * 40 + "\n"
    
    # Main prompt
    prompt = f"""You are analyzing features from a university student dropout prediction dataset. 

{context_section}

TASK: Evaluate ONLY the feature marked as "[CURRENT ROW TO EVALUATE]" above. The other rows are provided as context to help you understand patterns, relationships, and the overall dataset structure.

ANALYSIS GUIDELINES:
- Consider how this feature relates to the surrounding features
- Look for patterns in data types, cardinality, and missing values
- Consider the logical flow and relationships between consecutive features
- Use the context to make more informed decisions about derivation, modeling utility, and transformations

You must provide exactly 13 lines in the following format. Each line must contain exactly one colon (:) separating the field name from the value.

Required format (copy exactly, replace [...] with your answers):

Data Source: [Brief description of where this data likely comes from]
Is Derived?: [Yes/No]
Is Derived? Reason: [Brief specific explanation considering context]
Used in Modeling?: [Yes/No/Dropped]
Used in Modeling? Reason: [Brief specific explanation considering context]
Transformation Needed?: [Yes/No]
Transformation Needed? Reason: [Brief specific explanation considering context]
Time-Dependent?: [Yes/No]
Time-Dependent? Reason: [Brief specific explanation considering context]
Bucketed Version Exists?: [Yes/No]
Bucketed Version Exists? Reason: [Brief specific explanation considering context]
Interactional Potential?: [Yes/No]
Interactional Potential? Reason: [Brief specific explanation considering context]

CRITICAL REQUIREMENTS:
- Answer exactly Yes/No (or Yes/No/Dropped for modeling question)
- Keep reasons under 25 words and specific to this feature
- Consider the contextual relationships with nearby features
- No markdown, bullets, or extra formatting
- No additional commentary or explanations
- Must be exactly 13 lines total
- Focus ONLY on the feature marked with ">>>"
"""
    
    return prompt

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
        "max_tokens": 600,   # Slightly higher for contextual reasoning
        "top_p": 0.9
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=45)
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
    
    if len(valid_lines) != 13:
        logger.warning(f"Row {row_index} ({column_name}): Expected 13 lines, got {len(valid_lines)}")
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
                logger.warning(f"Row {row_index}: Unexpected key '{key}' - attempting to match")
                # Try to match similar keys
                for expected_field in EXPECTED_FIELDS:
                    if expected_field.lower() in key.lower() or key.lower() in expected_field.lower():
                        entry[expected_field] = value
                        logger.info(f"Row {row_index}: Matched '{key}' to '{expected_field}'")
                        break
                
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
        if field in entry and not entry[field].startswith("ERROR:"):
            answer = entry[field].upper()
            if answer not in ['YES', 'NO']:
                logger.warning(f"Row {row_index} ({column_name}): Invalid answer for '{field}': {entry[field]}")
    
    # Special validation for modeling field
    if "Used in Modeling?" in entry and not entry["Used in Modeling?"].startswith("ERROR:"):
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
    missing_columns = [col for col in CONTEXT_COLUMNS if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return
    
    # Check if output columns already exist (for resuming)
    existing_fields = [field for field in EXPECTED_FIELDS if field in df.columns]
    if existing_fields:
        logger.info(f"Found existing fields: {existing_fields}")
        # Determine starting point for resume
        start_index = 0
        for i, row in df.iterrows():
            if pd.isna(row.get('Data Source', '')) or row.get('Data Source', '') == "":
                start_index = i
                break
        logger.info(f"Resuming from row {start_index}")
    else:
        start_index = 0
        # Add empty columns for the fields we'll fill
        for field in EXPECTED_FIELDS:
            df[field] = ""
    
    responses = []
    parsed_entries = []
    
    # Process each row with context
    for index in range(start_index, len(df)):
        row = df.iloc[index]
        column_name = row['Column Name']
        logger.info(f"Processing row {index+1}/{len(df)}: {column_name} (with context window)")
        
        # Generate contextual prompt
        prompt = generate_contextual_prompt(df, index)
        
        # Log prompt size for monitoring
        token_estimate = estimate_token_count(prompt)
        logger.debug(f"Prompt token estimate: {token_estimate}")
        
        response = ask_deepseek(prompt)
        responses.append(response if response else "")
        
        parsed_entry = parse_response(response, index, column_name)
        parsed_entries.append(parsed_entry)
        
        # Update the dataframe with new values
        for field, value in parsed_entry.items():
            df.at[index, field] = value
        
        # Save checkpoint every 5 rows (more frequent due to context processing)
        if (index + 1) % 5 == 0:
            save_checkpoint(df, f"checkpoint_contextual_{index+1}.xlsx")
        
        time.sleep(REQUEST_DELAY)
    
    # Create final output
    try:
        df.to_excel(OUTPUT_PATH, index=False)
        logger.info(f"✅ Final output saved to {OUTPUT_PATH}")
        
        # Generate summary report
        generate_summary_report(df, len(df), start_index)
        
    except Exception as e:
        logger.error(f"Failed to create final output: {e}")

def generate_summary_report(df: pd.DataFrame, total_rows: int, start_index: int) -> None:
    """Generate a summary report of the processing results."""
    processed_rows = total_rows - start_index
    error_count = 0
    
    for field in EXPECTED_FIELDS:
        if field in df.columns:
            error_count += df[field].str.contains("ERROR:", na=False).sum()
    
    # Count completion rate
    completed_rows = 0
    for i in range(start_index, total_rows):
        if df.iloc[i].get('Data Source', '') and not str(df.iloc[i].get('Data Source', '')).startswith('ERROR:'):
            completed_rows += 1
    
    logger.info(f"""
    📊 CONTEXTUAL PROCESSING SUMMARY:
    - Total rows in dataset: {total_rows}
    - Rows processed in this run: {processed_rows}
    - Successfully completed rows: {completed_rows}
    - Total fields expected: {processed_rows * len(EXPECTED_FIELDS)}
    - Fields with errors: {error_count}
    - Success rate: {((processed_rows * len(EXPECTED_FIELDS) - error_count) / (processed_rows * len(EXPECTED_FIELDS)) * 100):.1f}%
    - Context window size used: {CONTEXT_WINDOW_SIZE} rows before/after
    """)

if __name__ == "__main__":
    main()
