# scripts/00_generate_dictionary_openai.py
import pandas as pd
import os
from openai import OpenAI  # Requires 'pip install openai'
from dotenv import load_dotenv # Requires 'pip install python-dotenv'
import time
import json # To potentially parse structured AI responses

# --- Configuration ---
RAW_DATA_PATH = 'data/survey/survey_raw.csv'
OUTPUT_DICT_PATH = 'data/survey/ai_generated_dictionary_detailed.csv'
# List known Qualtrics metadata columns/prefixes to exclude (adjust as needed)
QUALTRICS_METADATA_VARS = [
    'StartDate', 'EndDate', 'Status', 'IPAddress', 'Progress',
    'Duration (in seconds)', 'Finished', 'RecordedDate', 'ResponseId',
    'RecipientLastName', 'RecipientFirstName', 'RecipientEmail',
    'ExternalReference', 'LocationLatitude', 'LocationLongitude',
    'DistributionChannel', 'UserLanguage',
    'QID*', # Often internal Qualtrics IDs
    'SC*', # Often scoring categories
    # Add any other specific metadata columns you see in your file
    'Q46', 'Q47', # Assuming these are consent/location check from previous exploration
    'demo_anythingelse', 'Q44', 'Q45', 'Q46.1' # Other non-variable items
]
# Variables ending with these suffixes are often not primary data
SUFFIXES_TO_EXCLUDE = ('_TEXT', 'fu', '_Click Count', '_First Click', '_Last Click', '_Page Submit')
MAX_SAMPLE_VALUES = 5 # How many unique examples to send to AI

# --- Load API Key ---
load_dotenv() # Load variables from .env file into environment
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("ERROR: OPENAI_API_KEY not found in environment variables.")
    print("Please create a .env file in the project root with the line:")
    print("OPENAI_API_KEY='your_actual_api_key_here'")
    exit()

# Initialize OpenAI client
try:
    client = OpenAI(api_key=api_key)
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Make sure you have the correct API key and the library is installed.")
    exit()

# --- Enhanced Placeholder for OpenAI API Call ---
def generate_dictionary_entry(variable_name: str, question_text: str, sample_values: list, client: OpenAI) -> dict:
    """
    Uses OpenAI API to generate a dictionary entry including description, type, and coding.
    *** This is a placeholder - You MUST implement the actual API call logic below. ***
    """
    print(f"Processing variable: {variable_name}...") # Progress indicator
    entry = {
        'Description': 'Placeholder: Description error',
        'Type': 'Placeholder: Type error',
        'Coding': 'Placeholder: Coding error'
    }

    # ---vvv--- USER IMPLEMENTATION REQUIRED ---vvv---
    try:
        # 1. Construct the prompt requesting structured output (e.g., JSON or specific format)
        sample_values_str = ", ".join(map(str, sample_values))
        prompt = f"""
        Analyze the following survey variable to create a data dictionary entry:
        Variable Name: `{variable_name}`
        Full Question Text: "{question_text}"
        Sample Values: [{sample_values_str}]

        Based on this information, provide:
        1. Concise Description: (Summarize what the variable measures, ~1-2 sentences)
        2. Variable Type: (Choose ONE: Categorical, Numeric, Ordinal/Likert, Text, Other)
        3. Coding/Values: (Explain the meaning of the values, e.g., 1=Strongly Disagree...5=Strongly Agree, or list key categories. Use N/A if not applicable or too many unique values for sample.)

        Return the response ONLY in JSON format like this:
        {{"Description": "Your description here", "Type": "Your inferred type here", "Coding": "Your explanation of coding here or N/A"}}
        """

        # 2. Make the API call (ChatCompletions recommended for structured output)
        # Consult current OpenAI documentation. Use features like 'response_format' if available.
        # Example structure (check docs!):
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Or another suitable model
            messages=[
                {"role": "system", "content": "You are an assistant creating structured data dictionary entries in JSON format."},
                {"role": "user", "content": prompt}
            ],
            # response_format={ "type": "json_object" }, # If supported by model/API version
            max_tokens=150,
            temperature=0.2
        )
        response_content = response.choices[0].message.content

        # 3. Extract and Parse the AI response (assuming JSON, possibly wrapped)
        try:
            # Attempt to find the JSON object within the response string
            json_start = response_content.find('{')
            json_end = response_content.rfind('}') + 1 # rfind finds the last occurrence

            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_string = response_content[json_start:json_end]
                # Attempt to parse the extracted JSON string
                ai_results = json.loads(json_string)
                entry['Description'] = ai_results.get('Description', 'Error: Missing Description')
                entry['Type'] = ai_results.get('Type', 'Error: Missing Type')
                entry['Coding'] = ai_results.get('Coding', 'Error: Missing Coding')
            else:
                 # Handle cases where {} are not found or invalid range
                 print(f"  ERROR: Could not find valid JSON object delimiters in response for {variable_name}: {response_content}")
                 entry['Description'] = f"Error: Could not extract JSON - {response_content}"
                 entry['Type'] = "Parse Error (No JSON found)"
                 entry['Coding'] = "Parse Error (No JSON found)"

        except json.JSONDecodeError as json_err:
            # Handle cases where the extracted string is not valid JSON
            print(f"  ERROR: Failed to parse extracted JSON for {variable_name}: {json_string}. Error: {json_err}") # Note: json_string might not be defined if extraction failed before this point
            entry['Description'] = f"Error: Invalid JSON - {response_content}" # Log original response on error
            entry['Type'] = "Parse Error (Invalid JSON)"
            entry['Coding'] = "Parse Error (Invalid JSON)"
        except Exception as parse_e:
            # Handle other potential errors during parsing
            print(f"  ERROR: Unexpected error parsing response for {variable_name}: {parse_e}")
            entry['Description'] = f"Error: Unexpected parsing error - {parse_e}"
            entry['Type'] = "Parse Error (Unexpected)"
            entry['Coding'] = "Parse Error (Unexpected)"

        # Add delay
        time.sleep(1)

    except Exception as e:
        print(f"  ERROR calling/processing OpenAI API for {variable_name}: {e}")
        # Keep default error messages in entry dict
    # ---^^^--- USER IMPLEMENTATION REQUIRED ---^^^---

    return entry

# --- Main Script Logic ---
def main():
    try:
        # Read variable names (row 1)
        df_names = pd.read_csv(RAW_DATA_PATH, nrows=0)
        variable_names = df_names.columns.tolist()

        # Read question texts (row 2)
        df_questions = pd.read_csv(RAW_DATA_PATH, header=None, nrows=1, skiprows=1)
        question_texts = df_questions.iloc[0].tolist()

        # Create name -> question map
        if len(variable_names) == len(question_texts):
            name_question_map = dict(zip(variable_names, question_texts))
        else:
            print("ERROR: Number of variable names and question texts do not match.")
            exit()

        # Read the actual data (rows 3+), skipping the question row (row index 1)
        df_data = pd.read_csv(RAW_DATA_PATH, header=0, skiprows=[1])

    except FileNotFoundError:
        print(f"ERROR: Raw data file not found at {RAW_DATA_PATH}")
        exit()
    except Exception as e:
        print(f"Error reading raw data file: {e}")
        exit()

    # --- Filter Variables (REMOVED FOR EXHAUSTIVE DICTIONARY) ---
    # valid_variables = []
    # for var_name in df_data.columns:
    #     # Clean variable name (handle potential Qualtrics suffixes like '-1')
    #     clean_var_name = str(var_name).split('-')[0].strip()
    # 
    #     # Check against exact metadata list
    #     if clean_var_name in QUALTRICS_METADATA_VARS:
    #         print(f"Skipping metadata variable (exact match): {var_name}")
    #         continue
    # 
    #     # Check against suffixes
    #     if isinstance(clean_var_name, str) and clean_var_name.endswith(SUFFIXES_TO_EXCLUDE):
    #          print(f"Skipping variable due to suffix: {var_name}")
    #          continue
    # 
    #     # Check against metadata prefixes (like QID)
    #     is_metadata_prefix = False
    #     for prefix in QUALTRICS_METADATA_VARS:
    #         if prefix.endswith('*') and clean_var_name.startswith(prefix[:-1]):
    #             print(f"Skipping metadata variable (prefix match): {var_name}")
    #             is_metadata_prefix = True
    #             break
    #     if is_metadata_prefix:
    #         continue
    # 
    #     valid_variables.append(var_name) # Store original name for data access
    
    # Use all variables from the data file header directly
    all_variables = df_data.columns.tolist()

    # --- Generate Dictionary Entries ---
    data_dictionary = []
    # for var_name in valid_variables: # Iterate through ALL variables now
    for var_name in all_variables:
        question_text = name_question_map.get(var_name, "Error: Question text not found in map")

        # Handle cases where question text is missing or not a string
        if pd.isna(question_text) or not isinstance(question_text, str) or not question_text.strip():
            print(f"  WARNING: Invalid/missing question text for {var_name}. Skipping AI call.")
            ai_results = {
                'Description': f"Description unavailable (Missing/invalid question text: {question_text})",
                'Type': 'Unknown',
                'Coding': 'Unknown'
            }
            sample_values = [] # No samples needed if question is bad
        else:
            ai_results = None # Initialize ai_results before try/except
            # Get unique sample values
            try:
                sample_values = df_data[var_name].dropna().unique()[:MAX_SAMPLE_VALUES].tolist()
            except KeyError:
                 print(f"  ERROR: Column {var_name} not found in data frame. Skipping.")
                 continue # Skip this variable
            except Exception as e:
                 print(f"  ERROR getting samples for {var_name}: {e}. Skipping AI call.")
                 sample_values = [] # Can't get samples, proceed without them for AI?
                 # Decide if you want to call AI even without samples or skip
                 ai_results = {
                    'Description': f"Description error (Could not get samples: {e})",
                    'Type': 'Unknown',
                    'Coding': 'Unknown'
                 }
                 # If you want to try AI without samples, comment out the above and uncomment below
                 # ai_results = generate_dictionary_entry(var_name, question_text, [], client)

            # Only call AI if we have valid question text (and optionally samples)
            if ai_results is None: # Check if ai_results was set in sample error handling
                ai_results = generate_dictionary_entry(var_name, question_text, sample_values, client)

        # Get one example value directly from data
        try:
            example_value = df_data[var_name].dropna().iloc[0] if not df_data[var_name].dropna().empty else 'N/A'
        except Exception as e:
            example_value = f"Error: {e}"

        data_dictionary.append({
            'Variable': var_name,
            'Description': ai_results.get('Description', 'Error'),
            'Type': ai_results.get('Type', 'Error'),
            'Coding': ai_results.get('Coding', 'Error'),
            'Example Value': example_value
        })

    # --- Save Output ---
    df_dict = pd.DataFrame(data_dictionary)
    try:
        df_dict.to_csv(OUTPUT_DICT_PATH, index=False)
        print(f"\nAI-generated detailed data dictionary saved to {OUTPUT_DICT_PATH}")
    except Exception as e:
        print(f"Error saving detailed data dictionary: {e}")

if __name__ == "__main__":
    main() 