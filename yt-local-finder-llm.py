import streamlit as st
import pandas as pd
import time
import json
import re
from youtubesearchpython import VideosSearch
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
# import openai # Remove or comment out openai import
import ollama  # Import Ollama library
import requests # To check Ollama connection

# --- Configuration ---
MAX_VIDEOS_TO_CHECK = 5 # Keep lower for local model processing time
# LLM_MODEL = "gpt-3.5-turbo" # Remove or comment out OpenAI model
OLLAMA_MODEL = "llama3.1:latest" # <<< Your pulled Ollama model name
OLLAMA_HOST = "http://localhost:11434" # Default Ollama host
TRANSCRIPT_LANGUAGES = ['en', 'en-IN', 'hi']
MAX_VIDEOS_TO_FETCH = MAX_VIDEOS_TO_CHECK * 3

# --- Helper Functions ---

# Function to get OpenAI API key from Streamlit secrets
# def get_openai_key():
#     try:
#         return st.secrets["OPENAI_API_KEY"]
#     except KeyError:
#         st.error("OpenAI API key not found! Please add it to your Streamlit secrets (.streamlit/secrets.toml).")
#         st.stop() # Halt execution if key is missing

# # Initialize OpenAI client (do this once)
# openai.api_key = get_openai_key()

# --- Ollama Health Check ---
@st.cache_resource(ttl=30) # Cache check result for 30 seconds
def check_ollama_connection():
    """Checks if the Ollama server is running."""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags") # Simple endpoint check
        response.raise_for_status() # Raise error for bad status codes (4xx or 5xx)
        # Optionally check if the specific model is available
        models = response.json().get('models', [])
        model_tags = [m['name'] for m in models]
        if OLLAMA_MODEL not in model_tags:
             full_model_name_no_tag = OLLAMA_MODEL.split(':')[0]
             if any(m.startswith(full_model_name_no_tag + ':') for m in model_tags):
                 # If a variant exists but not the exact tag, it's probably fine
                 return True, None
             else:
                return False, f"Model '{OLLAMA_MODEL}' not found in Ollama. Available models: {', '.join(model_tags)}. Pull it using `ollama pull {OLLAMA_MODEL}`."
        return True, None # Connection and model OK
    except requests.exceptions.ConnectionError:
        return False, f"Ollama server not reachable at {OLLAMA_HOST}. Make sure Ollama is running."
    except requests.exceptions.RequestException as e:
        return False, f"Error connecting to Ollama: {e}"
    except Exception as e:
        return False, f"An unexpected error occurred checking Ollama: {e}"

@st.cache_data(ttl=3600) # Cache YouTube search results
def search_youtube(query, max_results):
    """Searches YouTube, filters for recent videos (<= 10 days), and returns details."""
    try:
        # Initial search - potentially fetches more than needed if max_results is high
        videos_search = VideosSearch(query, limit=max_results * 2) # Fetch more initially to increase chances after filtering
        results = videos_search.result()['result']
        
        # --- Filtering Logic Added Here ---
        filtered_results = []
        if results: # Proceed only if we got some results
            for video in results:
                published_time_str = video.get('publishedTime')
                if not published_time_str:
                    continue # Skip if no time info

                published_time_str = published_time_str.lower()
                is_within_10_days = False

                try:
                    # Check for hours or minutes - definitely recent enough
                    if 'hour' in published_time_str or 'minute' in published_time_str:
                        is_within_10_days = True
                    # Check for days
                    elif 'day' in published_time_str:
                        num_match = re.search(r'(\d+)', published_time_str)
                        if num_match:
                            days = int(num_match.group(1))
                            if days <= 10:
                                is_within_10_days = True
                    # Check for "a week ago" or "1 week ago"
                    elif 'week' in published_time_str:
                         # Simplistic check: only allow singular 'week'
                         # Assumes formats like "1 week ago" or "a week ago"
                         # Excludes "2 weeks ago" etc.
                         num_match = re.search(r'(\d+)', published_time_str)
                         number = int(num_match.group(1)) if num_match else 1
                         if number == 1:
                              is_within_10_days = True # 7 days is <= 10 days

                    # Anything mentioning "months" or "years" is excluded by default
                except Exception as e:
                    # Log parsing errors if needed, but don't stop processing
                    # print(f"Could not parse publishedTime '{video.get('publishedTime')}': {e}")
                    pass # Ignore videos with unparseable time strings

                if is_within_10_days:
                    # Ensure essential keys exist before adding
                    if all(k in video for k in ['id', 'title', 'link']):
                         filtered_results.append(video)
                         # Optional: Stop adding if we reached the original max_results limit
                         # if len(filtered_results) >= max_results:
                         #     break

        # Return only the filtered list, capped at the original max_results requested
        return filtered_results[:max_results]

    except Exception as e:
        st.error(f"Error searching YouTube: {e}")
        return []

@st.cache_data(ttl=86400) # Cache transcripts
def get_video_transcript(video_id):
    # (Same as before, maybe slightly adjusted logging)
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = None
        # Try manual first, then generated for specified languages
        for lang in TRANSCRIPT_LANGUAGES:
             try:
                 transcript = transcript_list.find_manually_created_transcript([lang])
                 # st.info(f"Found manual transcript ({lang}) for {video_id}")
                 break
             except NoTranscriptFound:
                 continue # Try next language
        if not transcript:
            for lang in TRANSCRIPT_LANGUAGES:
                try:
                    transcript = transcript_list.find_generated_transcript([lang])
                    # st.info(f"Found generated transcript ({lang}) for {video_id}")
                    break
                except NoTranscriptFound:
                     continue # Try next language

        if not transcript:
            st.warning(f"No suitable transcript found ({'/'.join(TRANSCRIPT_LANGUAGES)}) for video ID: {video_id}")
            return None

        full_transcript_segments = transcript.fetch()
        # Combine segments into a single text block for the LLM
        full_transcript_text = " ".join([segment.text for segment in full_transcript_segments])
        return full_transcript_text

    except TranscriptsDisabled:
        st.warning(f"Transcripts disabled for video ID: {video_id}")
        return None
    except Exception as e:
        # st.error(f"Error fetching transcript for video ID {video_id}: {e}") # Less verbose error for missing transcripts
        st.warning(f"Could not fetch transcript for video ID {video_id}.")
        return None

def create_llm_prompt(query, transcript_text):
    """Creates the detailed prompt for the LLM (Llama 3 focus)."""
    parts = query.split(" in ")
    product_desc = parts[0]
    location_context = parts[1] if len(parts) > 1 else "the specified location"

    # Llama 3 works well with clear instructions and JSON format requests in the prompt
    prompt = f"""
You are an expert information extractor analyzing a YouTube video transcript.
The user wants information about: "{product_desc}" in/around "{location_context}".

**Task:** Analyze the transcript below. Extract ONLY the following details IF explicitly mentioned and relevant:

1.  **Specific Items/Products Mentioned:** Instances of "{product_desc}" (e.g., specific car models).
2.  **Price:** Asking prices/ranges for identified items (note currency if possible, assume INR if context implies India).
3.  **Key Features/Specifications:** Model year, km driven, fuel, transmission, variant, condition, color, etc. for relevant items.
4.  **Availability/Location Hints:** Mentions of "for sale," "available," or specific dealership names/locations *within* "{location_context}".
5.  **Contact Information:** Phone numbers, addresses, calls to action *if mentioned*.

**Input Transcript:**
{transcript_text[:15000]}
**Output Instructions:**
*   Focus ONLY on relevant info for "{product_desc}" in "{location_context}".
*   Group findings per specific item if possible.
*   For each finding, include a short `context_snippet` from the transcript.
*   If no relevant info is found, respond with an empty JSON list `[]`.
*   **Format your entire response ONLY as a valid JSON list of objects.** Each object must have keys: `item_description` (string), `finding_type` (string), `value` (string), `context_snippet` (string).

**Example JSON Output Structure:**
```json
[
  {{
    "item_description": "2017 Honda City VX",
    "finding_type": "Price",
    "value": "₹ 7.2 Lakh Asking",
    "context_snippet": "...this 2017 Honda City VX automatic, they are asking for 7.2 lakh rupees..."
  }}
]
Ensure the output starts with [ and ends with ]. Do not add any text before or after the JSON list.
Do not copy the example directly. Analyze the transcript and generate a unique response.
"""
    return prompt

# @st.cache_data(ttl=3600, show_spinner=False) # Cache LLM results based on transcript & query
# def analyze_transcript_with_llm(query, transcript_text):
#     """Sends transcript to LLM and parses the structured response."""
#     if not transcript_text:
#         return []

#     prompt = create_llm_prompt(query, transcript_text)

#     try:
#         response = openai.chat.completions.create(
#             model=LLM_MODEL,
#             messages=[
#                 {"role": "system", "content": "You are an expert information extractor specializing in analyzing video transcripts for product details."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.2, # Lower temperature for more factual extraction
#             max_tokens=1000, # Adjust as needed, affects cost and completeness
#             response_format={ "type": "json_object" } # Request JSON output directly if using newer models/API versions
#         )

#         # content = response.choices[0].message.content
#         content = response.choices[0].message.content.strip()

#         # Debug: Print raw LLM response
#         # print("--- LLM Raw Response ---")
#         # print(content)
#         # print("------------------------")


#         # The response should ideally be the JSON string directly
#         # Sometimes it might be wrapped in ```json ... ```, try to extract
#         if content.startswith("```json"):
#             content = content.strip("```json").strip("`").strip()

#         # Parse the JSON response
#         try:
#             # Look for the start of the list '[' or object '{'
#             json_start_index = -1
#             list_start = content.find('[')
#             obj_start = content.find('{')

#             if list_start != -1 and (obj_start == -1 or list_start < obj_start):
#                 json_start_index = list_start
#             elif obj_start != -1:
#                 json_start_index = obj_start

#             if json_start_index == -1:
#                 st.warning("LLM response did not contain detectable JSON start.")
#                 print(f"LLM Response (no JSON start): {content}") # Debugging
#                 return []


#             # Find the matching closing bracket/brace (handle potential nesting crudely)
#             json_end_index = -1
#             if json_start_index == list_start:
#                 json_end_index = content.rfind(']')
#             else: # object start
#                 json_end_index = content.rfind('}')

#             if json_end_index == -1 or json_end_index < json_start_index:
#                 st.warning("LLM response JSON structure seems incomplete (missing end bracket/brace).")
#                 print(f"LLM Response (incomplete JSON): {content}") # Debugging
#                 return []


#             json_string = content[json_start_index : json_end_index + 1]


#             #print(f"Attempting to parse JSON: {json_string}") # Debugging
#             extracted_data = json.loads(json_string)

#             # Ensure it's a list, even if the LLM returned a single object incorrectly
#             if isinstance(extracted_data, dict):
#                 extracted_data = [extracted_data] # Wrap single object in a list

#             # Validate structure minimally
#             validated_data = []
#             if isinstance(extracted_data, list):
#                 for item in extracted_data:
#                     if isinstance(item, dict) and all(k in item for k in ['finding_type', 'value', 'context_snippet']):
#                         # Add item_description if missing (use a default)
#                         if 'item_description' not in item:
#                             item['item_description'] = "Unknown Item"
#                         validated_data.append(item)
#                     else:
#                         st.warning(f"LLM returned an item with unexpected structure: {item}")
#                         print(f"LLM returned item with unexpected structure: {item}") # Debugging
#             return validated_data

#         except json.JSONDecodeError as json_err:
#             st.error(f"Failed to parse LLM response as JSON.")
#             st.error(f"Error: {json_err}")
#             st.error(f"LLM Raw Response was:\n```\n{content}\n```")
#             return []

#     except openai.APIError as e:
#         st.error(f"OpenAI API Error: {e}")
#         return []
#     except Exception as e:
#         st.error(f"An unexpected error occurred during LLM analysis: {e}")
#         return []

@st.cache_data(ttl=3600, show_spinner=False) # Cache Ollama results
def analyze_transcript_with_ollama(query, transcript_text):
    """Sends transcript to local Ollama model and parses the structured response."""
    if not transcript_text:
        return []

    # Generate the user prompt content
    user_prompt_content = create_llm_prompt(query, transcript_text)

    try:
        # Use ollama.chat for better control with system/user roles
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    'role': 'system',
                    # Define the expected behavior clearly for Llama 3
                    'content': 'You are an expert information extractor. Your goal is to analyze the user-provided transcript based on their query and return ONLY a valid JSON list of findings as specified in the user prompt. Do not copy the example JSON directly. Use it only as a reference for formatting. Analyze the transcript and generate a unique response based on the input. Do not include any introductory text, explanations, or markdown formatting around the JSON.',
                },
                {
                    'role': 'user',
                    'content': user_prompt_content,
                },
            ],
            format='json',  # Instruct Ollama to attempt generating valid JSON
            options={       # Optional: Adjust generation parameters if needed
                'temperature': 0.2,
                # 'num_predict': 1024 # Limit output length if necessary
            }
        )

        content = response['message']['content'].strip()

        # --- JSON Parsing Logic (mostly the same as before) ---
        # Debug: Print raw response
        # print("--- Ollama Raw Response ---")
        # print(content)
        # print("---------------------------")

        if not content:
            st.warning("Ollama returned an empty response.")
            return []

        # Basic cleanup: sometimes models still wrap in markdown
        if content.startswith("```json"):
            content = content.strip("```json").strip("`").strip()
        elif content.startswith("```"):
            content = content.strip("```").strip()

        # Attempt to parse the JSON
        try:
            # Find the start, handle potential leading text if model ignored instruction
            json_start = content.find('[')
            if json_start == -1:
                # Maybe it returned a single object instead of a list? Less likely with prompt.
                json_start = content.find('{')
                if json_start == -1:
                    st.warning("Ollama response did not contain detectable JSON start '[' or '{'.")
                    print(f"Ollama Response (no JSON start): {content}")
                    return []

            # Find the corresponding end bracket/brace (crude matching)
            if content[json_start] == '[':
                json_end = content.rfind(']')
            else:
                json_end = content.rfind('}')

            if json_end == -1 or json_end < json_start:
                st.warning("Ollama response JSON structure incomplete (missing end bracket/brace).")
                print(f"Ollama Response (incomplete JSON): {content}")
                return []

            json_string = content[json_start : json_end + 1]

            # print(f"Attempting to parse JSON from Ollama: {json_string}") # Debugging
            extracted_data = json.loads(json_string)

            # Validate structure (same as before)
            if isinstance(extracted_data, dict): extracted_data = [extracted_data] # Wrap single object
            validated_data = []
            if isinstance(extracted_data, list):
                for item in extracted_data:
                    if isinstance(item, dict) and all(k in item for k in ['finding_type', 'value', 'context_snippet']):
                        if 'item_description' not in item: item['item_description'] = "Unknown Item"
                        validated_data.append(item)
                    else:
                        st.warning(f"Ollama returned item with unexpected structure: {item}")
                        print(f"Ollama returned item with unexpected structure: {item}") # Debug
            return validated_data

        except json.JSONDecodeError as json_err:
            st.error(f"Failed to parse Ollama response as JSON.")
            st.error(f"Error: {json_err}")
            st.error(f"Ollama Raw Response Content was:\n```\n{content}\n```")
            return []
        except Exception as e: # Catch other potential parsing errors
            st.error(f"Error processing Ollama JSON response: {e}")
            st.error(f"Ollama Raw Response Content was:\n```\n{content}\n```")
            return []

    # Specific Ollama connection errors (using requests check earlier helps)
    except requests.exceptions.ConnectionError: # Catch if check failed somehow
        st.error(f"Ollama server connection failed at {OLLAMA_HOST}. Is it running?")
        return []
    except Exception as e: # Catch errors from ollama library or other issues
        st.error(f"An error occurred during Ollama analysis: {e}")
        # Potentially log the full exception traceback here for debugging
        # import traceback
        # st.exception(e)
        return []

#--- Streamlit UI ---
st.set_page_config(page_title="Local Product Finder (Ollama)", layout="wide")
st.title(" Finde Local Product Info from YouTube (Ollama Powered)")
st.markdown(f"""
Enter product and location. The app uses a locally running Ollama model ({OLLAMA_MODEL})
to analyze YouTube transcripts for prices, features, etc.

Prerequisites:

Ollama must be running locally.

The model {OLLAMA_MODEL} must be pulled (ollama pull {OLLAMA_MODEL}).

Disclaimer: AI analysis can make mistakes. Always verify details by watching the video. Performance depends on your local hardware.
""")

#--- Check Ollama Status ---
is_ollama_ok, error_message = check_ollama_connection()
if not is_ollama_ok:
    st.error(f"Ollama Status Check Failed: {error_message}")
    st.warning("The app requires a running Ollama instance with the specified model to function.")
    # You could disable the form or stop execution here if preferred
    # st.stop()
else:
    st.success(f"Ollama connection successful. Using model: {OLLAMA_MODEL}")

#--- Input Form ---
with st.form("search_form"):
    product_name = st.text_input("Product Name", placeholder="e.g., second hand Swift, used iPhone 12")
    location = st.text_input("Location", placeholder="e.g., Delhi, Mumbai, Bangalore")
    submitted = st.form_submit_button("Search & Analyze with AI")

#--- Processing and Output ---
if submitted and product_name and location:
    # Ensure Ollama is still OK before proceeding (in case it stopped)
    is_ollama_ok, error_message = check_ollama_connection()
    if not is_ollama_ok:
        st.error(f"Ollama Status Check Failed: {error_message}")
        st.stop() # Stop if Ollama isn't ready when submitting
    # Construct a query that's good for YouTube search AND gives context to the LLM prompt later
    query = f"{product_name} in {location}"
    search_query_yt = f"{product_name} {location} sale"
    st.write(f"Searching YouTube for: '{search_query_yt}'...")

    videos = search_youtube(search_query_yt, MAX_VIDEOS_TO_FETCH)

    if not videos:
        st.warning("No relevant videos found. Try different keywords.")
        st.stop()
    
    # We might have fewer videos than MAX_VIDEOS_TO_FETCH if filtering was aggressive
    actual_videos_fetched = len(videos)
    st.write(f"Found {actual_videos_fetched} potential recent videos. Analyzing up to {MAX_VIDEOS_TO_CHECK} initially...")

    all_findings = [] # Use 'all_ollama_findings' or 'all_llm_findings' based on your version
    processed_video_count = 0
    videos_with_transcripts_found = 0 # *** CHANGE 2: Initialize transcript counter ***

    progress_bar = st.progress(0)
    status_text = st.empty()

    # *** CHANGE 3: Modify the main loop logic ***
    for i, video in enumerate(videos): # Iterate through *all* fetched videos

        # --- Condition to stop processing extra videos ---
        # If we are past the initial target count AND already found enough transcripts, stop early.
        if i >= MAX_VIDEOS_TO_CHECK and videos_with_transcripts_found >= 2:
            status_text.text(f"Found sufficient transcripts ({videos_with_transcripts_found}) within the first {i} videos. Stopping analysis.")
            time.sleep(1.5) # Let user see the message
            break # Stop processing further videos

        # --- Message if extending search ---
        if i == MAX_VIDEOS_TO_CHECK and videos_with_transcripts_found < 2:
            status_text.warning(f"Found only {videos_with_transcripts_found} transcript(s) in first {MAX_VIDEOS_TO_CHECK} videos. Checking remaining {actual_videos_fetched - i} videos...")
            time.sleep(2) # Pause so user notices the message

        video_id = video['id']
        video_title = video['title']
        video_link = video['link']

        # Update status based on current video index relative to total fetched
        # Use min(i+1, actual_videos_fetched) in case we break early but want progress to reach 100% eventually
        progress_val = (i + 1) / actual_videos_fetched
        progress_bar.progress(progress_val)
        status_text.text(f"Processing video {i+1}/{actual_videos_fetched}: {video_title[:50]}...")

        # 1. Get Transcript Text
        transcript_text = get_video_transcript(video_id)

        if transcript_text:
            # *** CHANGE 4: Increment counter ONLY if transcript exists ***
            videos_with_transcripts_found += 1
            status_text.text(f"Analyzing transcript for video {i+1} ({videos_with_transcripts_found} transcript(s) found so far)...")

            # 2. Analyze with Ollama/LLM (Use your specific function call)
            results = analyze_transcript_with_ollama(query, transcript_text) # OR analyze_transcript_with_llm(...)

            if results:
                # Add video context and append results...
                for finding in results:
                    finding['Video Title'] = video_title
                    finding['Video Link'] = video_link
                    finding['Video ID'] = video_id
                all_findings.extend(results) # Use the correct list name
        # else: # Transcript not found or error
        #     # No action needed here, counter wasn't incremented
        #     pass

        processed_video_count += 1 # Count how many loops were actually run

        # Optional: Add a small delay if using a local model to prevent UI freeze on heavy load
        # time.sleep(0.05)

    # --- Adjust Final Status Update ---
    status_text.success(f"Analysis complete. Processed {processed_video_count} videos. Found {videos_with_transcripts_found} with transcripts and {len(all_findings)} potential mentions.")
    progress_bar.empty()

    # --- Display results (Keep your existing display logic here) ---
    if not all_findings:
        st.info("Could not extract specific details... (Your existing message)")
    else:
        st.subheader("Ollama-Extracted Information:")
        st.markdown("**Note:** `Transcript Context` shows text near the finding. Use `Video Link` to verify.")

        findings_by_video = {}
        # (Grouping logic remains the same)
        for finding in all_findings:
            vid_id = finding['Video ID']
            if vid_id not in findings_by_video:
                findings_by_video[vid_id] = {'title': finding['Video Title'], 'link': finding['Video Link'], 'findings': []}
            findings_by_video[vid_id]['findings'].append(finding)

        for vid_id, video_data in findings_by_video.items():
            st.markdown(f"---")
            st.markdown(f"**Video:** [{video_data['title']}]({video_data['link']})")
            df_video = pd.DataFrame(video_data['findings'])
            df_display = df_video[['item_description', 'finding_type', 'value', 'context_snippet']]
            df_display = df_display.rename(columns={'item_description': 'Item', 'finding_type': 'Type', 'value': 'Details', 'context_snippet': 'Transcript Context'})
            st.table(df_display) # Use st.table for better auto-sizing

        with st.expander("Show Raw Extracted Data (JSON)"):
            st.json(all_findings)

elif submitted:
    st.warning("Please enter both Product Name and Location.")

#--- Footer/Info ---
st.markdown("---")
st.caption(f"Using local Ollama model: {OLLAMA_MODEL} | Ensure Ollama server is running.")