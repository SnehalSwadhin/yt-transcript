import pandas as pd
from youtube_transcript_api._api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.proxies import WebshareProxyConfig
import openai
import ollama
from groq import Groq
from google import genai
from google.genai import types
import os, re, io, json
from dotenv import load_dotenv
from logger import log
from thefuzz import fuzz
from database import initialize_db, is_video_processed, add_cars_to_db
from googleapiclient.discovery import build
from datetime import datetime, timedelta, timezone

load_dotenv()
DAILY_VIDEOS_LIMIT = 5
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY environment variable not set! Please create a .env file.")
youtube_api = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
llm_tool = None
proxy_username = os.getenv("WEBSHARE_PROXY_USERNAME")
proxy_password = os.getenv("WEBSHARE_PROXY_PASSWORD")

# Define the two prompts in our chain
prompt_1_template = """You are an expert data extractor. Your task is to read the following transcript, which could be in any language, and produce a summary **in English**.

The summary must follow these specific rules:
1. Identify every unique vehicle mentioned in the transcript.
2. For each vehicle, create a primary bullet point using its name (e.g., "* 2024 Hyundai Creta").
3. Under each primary bullet point, create a nested list of all its specifications.
4. Each specification in the nested list must use the format "Attribute: Value" (e.g., "- Price: ₹7.5 lakh", "- Color: Blue:").
5. Note that the "Price" attribute should be a numerical value
6. Add a final bullet point for "original_quote": A SHORT, unique, and EXACT quote from the original transcript that clearly identifies this specific car. **This quote MUST be in the original language of the transcript. and MUST be continuous**

---
Transcript to process:
{transcript}
"""

# We explicitly ask for a markdown table and "NA" for missing values.
# Adding "and nothing else" helps prevent the model from adding conversational text.
prompt_2_template = """Structure all of the information given in following summary in a tabular format.
- Create columns for the following attributes:
OEM | Model | Variant | Price | Colour | Odo | Year | Service record | Frame type | Transmission type | Fuel type | Number of owners | RTO  | City | Engine details | Feature details | Rating
- If a piece of data is not available for a specific car, just put NA.
- The output should be a clean markdown table and nothing else.
- Add a final column "original_quote" mentioned in the summary
- Convert the "Price" column to a single numeric value (e.g., ₹7.5 lakh should be converted to 750000).
- If there are no cars mentioned, do not output a table.
---
Summary:
{summary}
"""

prompt_3_template = """You are an expert contact information extractor. From the provided text, which includes a video description and parts of a transcript, extract the following details for the car dealer.

The details to extract are:
- "dealer_name": The name of the dealership or person selling the cars.
- "dealer_contact": A phone or mobile number. Extract only the number.
- "dealer_email": An email address.
- "dealer_website": A website URL.
- "dealer_location": The physical address or area mentioned.

Structure your output as a single JSON object. If a piece of information is not found, use `null` for that field's value.

Here is the text to analyze:
---
{contact_corpus}
---
"""

class LLMTool:
    """Handles all interactions with the Language Model API (OpenAI)."""

    def __init__(self, llm_model='gemini-pro'):
        """Initializes the LLM client."""
        self.llm_model = llm_model.lower()
        if self.llm_model == 'gemini-flash' or llm_model == 'gemini-pro':
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables.")
            self.gemini_client = genai.Client(api_key=api_key)
        elif self.llm_model == 'openai':
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables.")
            self.openai_client = openai.OpenAI(api_key=api_key)
        elif self.llm_model == 'groq':
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables.")
            self.groq_client = Groq(api_key=api_key)

        log.info(f"LLMTool: {self.llm_model} client initialized.")
        
    from typing import Optional

    def _make_request(self, system_prompt: str, user_prompt: str, is_json: bool = False, thinking_mode: bool = False, model: Optional[str] = None) -> str:
        """A private helper to make a generic chat completion request."""
        result = None
        selected_model = model if model else self.llm_model
        log.info(f"Making request to {selected_model} model with thinking mode: {thinking_mode}")
        try:
            if selected_model == "openai":
                response_1 = self.openai_client.chat.completions.create(
                    model="o4-mini", # Using a more advanced model can yield better results
                    reasoning_effort="medium" if thinking_mode else None,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"} if is_json else {"type": "text"}
                )
                result = response_1.choices[0].message.content
            elif selected_model == "gemini-pro":
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json" if is_json else "text/plain",
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=-1
                        )
                    ),
                )
                result = response.text
            elif selected_model == "gemini-flash":
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json" if is_json else "text/plain",
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=-1 if thinking_mode else 0,  # -1 for unlimited thinking time, 0 for no thinking
                        )
                    ),
                )
                result = response.text
            elif selected_model == "ollama":
                response = ollama.chat(
                    model="llama3.2:latest",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ], 
                    think=thinking_mode,
                    format="json" if is_json else None,
                    # options={       # Optional: Adjust generation parameters if needed
                    #     'temperature': 0.2,
                    #     # 'num_predict': 1024 # Limit output length if necessary
                    # }
                )
                result = response['message']['content'].strip()
            elif selected_model == "groq":
                response = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    reasoning_effort="default" if thinking_mode else "none",
                    response_format={"type": "json_object"} if is_json else {"type": "text"}
                )
                result = response.choices[0].message.content
            else:
                raise ValueError(f"Unsupported LLM model: {self.llm_model}")
            return result if result else "<No response from LLM>"
        except Exception as e:
            log.error(f"Error calling OpenAI API: {e}")
            return "<No response from LLM>"

def run_prompt_chain_openai(transcript):
    """
    Executes the two-step prompt chain to extract and structure car information.
    """
    if not llm_tool:
        raise ValueError("LLMTool is not initialized.")
    log.info("--- Step 1: Kicking off the first prompt (Summarization) ---")
    # Format the first prompt with the transcript
    summary = llm_tool._make_request(
        system_prompt="You are an expert assistant skilled at summarizing technical specifications from text.",
        user_prompt=prompt_1_template.format(transcript=transcript)
    )
    
    log.info("\n✅ Intermediate Summary from Prompt 1:\n")
    log.info(summary)
    
    log.info("\n--- Step 2: Kicking off the second prompt (Structuring) ---")
    
    # Format the second prompt with the summary from the first call
    structured_data_string = llm_tool._make_request(
        system_prompt="You are an expert data formatter. Your job is to convert text into clean markdown tables.",
        user_prompt=prompt_2_template.format(summary=summary),
        model="gemini-flash"
    )
    
    log.info("\n✅ Structured Markdown Table from Prompt 2:\n")
    log.info(structured_data_string)
    
    return structured_data_string

def parse_markdown_to_dataframe(llm_output_string: str) -> pd.DataFrame:
    """
    Parses a markdown table from a potentially messy LLM output string
    into a clean pandas DataFrame. It handles surrounding text and code fences.
    """
    log.info("\n--- Step 3: Parsing LLM output into a Pandas DataFrame ---")

    # Use regex to find the markdown table, including the optional ```markdown fence
    # re.DOTALL allows '.' to match newlines
    match = re.search(r"```(?:markdown)?\n([\s\S]*?)\n```", llm_output_string, re.DOTALL)
    
    if match:
        # If a fenced code block is found, use its content
        markdown_table = match.group(1).strip()
    else:
        # If no code block is found, fall back to finding contiguous lines
        # that look like a table.
        lines = llm_output_string.strip().split('\n')
        table_lines = [line for line in lines if line.strip().startswith('|') and line.strip().endswith('|')]
        if not table_lines:
             raise ValueError("Could not find a markdown table in the LLM output.")
        markdown_table = "\n".join(table_lines)
        
    log.info("\n✅ Successfully Extracted Markdown Table:\n")
    log.info(markdown_table)
    
    # Use io.StringIO to treat the string as a file for pandas
    # The markdown table uses '|' as a separator.
    # We skip the separator line (e.g., |---|---|) by filtering it out.
    lines = markdown_table.strip().split('\n')
    cleaned_lines = [line for line in lines if not set(line.strip()).issubset({'|', '-', ' ', ':'})]
    
    data_string = "\n".join(cleaned_lines)
    data = io.StringIO(data_string)
    
    # Read the data using the '|' separator
    df = pd.read_csv(data, sep='|', index_col=False)

    # Drop the empty columns created by the leading and trailing '|'
    df = df.drop(df.columns[[0, -1]], axis=1)

    # Strip whitespace from column names and all string cells
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    return df

def find_best_match_in_transcript(quote, structured_transcript, window_size=3, score_cutoff=85):
    """
    Finds the best match for a quote in a transcript, considering that the quote
    might span multiple chunks and might not be an exact match.

    Args:
        quote (str): The string to search for (from the LLM).
        structured_transcript (list): The list of {'text', 'start', ...} dicts.
        window_size (int): The number of consecutive chunks to combine for searching.
        score_cutoff (int): The minimum fuzzy match score (0-100) to consider a match valid.

    Returns:
        The starting chunk dictionary of the best match if the score is above cutoff, otherwise None.
    """
    if not quote or not structured_transcript:
        return None

    best_match_score = -1
    best_match_chunk = None

    chunks = structured_transcript
    
    # Iterate through each chunk as a potential starting point for a window
    for i in range(len(chunks)):
        # Create a sliding window of text
        window_chunks = chunks[i : i + window_size]
        combined_text = " ".join([c.text for c in window_chunks])

        # Use fuzzy matching to get a similarity score.
        # fuzz.partial_ratio is often good for finding a sub-string within a larger text block.
        # You can also experiment with fuzz.token_set_ratio.
        current_score = fuzz.partial_ratio(quote, combined_text)
        
        # If this window gives a better score, update our best match
        if current_score > best_match_score:
            best_match_score = current_score
            # The relevant chunk is the one at the start of the window
            best_match_chunk = chunks[i]

            # Optimization: If we find a perfect match, we can stop early.
            if best_match_score > 95:
                break
    
    # After checking all windows, decide if the best match is good enough
    if best_match_chunk is not None and best_match_score >= score_cutoff:
        log.info(f"Found match for '{quote}' with score {best_match_score} starting at {best_match_chunk.start}s.")
        return best_match_chunk
    else:
        log.info(f"No suitable match found for '{quote}'. Best score was {best_match_score}, which is below the cutoff of {score_cutoff}.")
        return None

def find_original_quote_in_transcript(search_quote, structured_transcript, video_id):
    """
    For each row in the DataFrame, find the original quote in the transcript
    and add its details in the dataframe.
    """
    start_chunk = find_best_match_in_transcript(
        quote=search_quote, 
        structured_transcript=structured_transcript,
        window_size=3,      # Look at 3 chunks at a time
        score_cutoff=80     # Require at least 85% similarity
    )
    if start_chunk:
        # If we found a match, return the start time and text
        return {
            "start_timestamp": start_chunk.start,
            "video_link": f"https://www.youtube.com/watch?v={video_id}&t={int(start_chunk.start)}s",
        }
    else:
        # If no match was found, return None
        return {
            "start_timestamp": None,
            "video_link": None,
        }

def create_contact_corpus(structured_transcript, description, num_chunks=25):
    """
    Creates a small, targeted text corpus for extracting contact info.

    Args:
        structured_transcript (list): The list of {'text', 'start'} dicts.
        description (str): The full text of the video description.
        num_chunks (int): The number of chunks to take from the start and end.

    Returns:
        str: A combined string of the most relevant text.
    """
    # Get the first N chunks of the transcript
    intro_chunks = structured_transcript[:num_chunks]
    intro_text = " ".join([chunk.text for chunk in intro_chunks])

    # Get the last N chunks of the transcript
    outro_chunks = structured_transcript[-num_chunks:]
    outro_text = " ".join([chunk.text for chunk in outro_chunks])

    # Combine all relevant text into one block
    corpus = f"""
--- VIDEO DESCRIPTION ---
{description}

--- TRANSCRIPT INTRO ---
{intro_text}

--- TRANSCRIPT OUTRO ---
{outro_text}
    """
    return corpus

def extract_dealer_details(contact_corpus):
    if not llm_tool:
        raise ValueError("LLMTool is not initialized.")
    log.info("--- Step 1: Kicking off the first prompt (Summarization) ---")
    response_content = llm_tool._make_request(
        system_prompt="You are a helpful assistant designed to output JSON. Your task is to extract dealer details from the provided text.",
        user_prompt=prompt_3_template.format(contact_corpus=contact_corpus),
        is_json=True
    )

    try:
        # We can directly parse it without any cleaning
        if response_content:
            dealer_info = json.loads(response_content)
            return dealer_info
        else:
            log.info("No dealer information found in the response.")
            return None
    except json.JSONDecodeError:
        # This block is now mostly for safety; it should rarely be triggered
        log.error("Model returned invalid JSON even in JSON mode.")
        return None

def run_full_scrape_pipeline_for_video(video):
    video_id = video.get('id')
    if not video_id:
        log.error("No video ID found in the search result.")
        return None
    structured_transcript = None
    transcript_text = ""
    log.info(f"Starting transcript extraction for video ID: {video_id}")

    try:
        if proxy_username == None or proxy_password == None:
            log.warning("Proxy credentials not found. Proceeding without proxy.")
            return None
        transcript_list = YouTubeTranscriptApi(proxy_config=WebshareProxyConfig(
            proxy_username=proxy_username,
            proxy_password=proxy_password,
        )).list(video_id=video_id)
        for transcript in transcript_list:
            transcript_type = "Auto-Generated" if transcript.is_generated else "Manually Created"
            log.info(f"Processing transcript: {transcript.language} ({transcript.language_code})")
            fetchhed = transcript.fetch()
            transcript_text = " ".join(segment.text for segment in fetchhed)
            log.info(
                f"- Language: {transcript.language}, "
                f"Code: {transcript.language_code}, "
                f"Type: {transcript_type}"
            )
            structured_transcript = fetchhed
            if not transcript.is_generated:
                break
    except TranscriptsDisabled:
        log.warning(f"Transcripts are disabled for video ID: {video_id}")
        return None
    except NoTranscriptFound:
        log.warning(f"No transcripts could be found for video ID: {video_id}")
        return None
    except Exception as e:
        log.error(f"An error occurred: {e}")
        return None
    
    markdown_output = run_prompt_chain_openai(transcript_text)

    if markdown_output is not None:
        df = parse_markdown_to_dataframe(markdown_output)
        df = df.replace(r'^\s*(n/a|nan|null|na|NA|NaN|Null|N/a|N/A)\s*$', None, regex=True)
        if df.empty or df['OEM'][0] == None:
            log.warning("No data was extracted from the markdown output.")
            return None
        df[['start_timestamp', 'video_link']] = df.apply(lambda x: find_original_quote_in_transcript(x['original_quote'], structured_transcript, video_id), axis=1, result_type="expand")
        df['video_id'] = video['id']
        df['video_title'] = video['title']
        df['published_at'] = video['published_at']
        df['channel_title'] = video['channel_title']
        df['timestamp'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

        contact_corpus = create_contact_corpus(structured_transcript, video.get('description', ''))
        if contact_corpus:
            dealer_info = extract_dealer_details(contact_corpus)
            if dealer_info:
                df['dealer_name'] = dealer_info.get('dealer_name', 'Unknown')
                df['dealer_contact'] = dealer_info.get('dealer_contact', 'Unknown')
                df['dealer_email'] = dealer_info.get('dealer_email', 'Unknown')
                df['dealer_website'] = dealer_info.get('dealer_website', 'Unknown')
                df['dealer_location'] = dealer_info.get('dealer_location', 'Unknown')
            else:
                log.warning("No dealer information extracted.")
                df[['dealer_name', 'dealer_contact', 'dealer_email', 'dealer_website', 'dealer_location']] = None
    else:
        log.error("No markdown output was generated from the prompt chain.")
        df = None
    return df

def search_recent_videos(query, days=1):
    """
    Searches YouTube for videos matching a query published in the last `days`.

    Args:
        query (str): The search term (e.g., "used cars bangalore").
        days (int): How many days back to search.

    Returns:
        list: A list of dictionaries, each containing video details.
    """
    # Calculate the 'publishedAfter' date in the required RFC 3339 format
    start_time = datetime.now(timezone.utc) - timedelta(days=days)
    published_after_str = start_time.isoformat()

    try:
        # This is the core API call
        request = youtube_api.search().list(
            q=query,
            part='snippet',      # 'snippet' includes title, description, publishedAt, etc.
            type='video',        # We only want videos, not channels or playlists
            order='date',        # Order by date to get the most recent ones first
            maxResults=DAILY_VIDEOS_LIMIT+10,       # Get up to 5 results (the max per page)
            publishedAfter=published_after_str,
            videoDuration='long'
        )
        
        response = request.execute()
        
        videos = []
        for item in response.get('items', []):
            video_id = item['id']['videoId']
            snippet = item['snippet']
            
            videos.append({
                'id': video_id,
                'title': snippet['title'],
                'published_at': snippet['publishedAt'],
                'description': snippet['description'],
                'channel_title': snippet['channelTitle']
            })
        
        request = youtube_api.search().list(
            q=query,
            part='snippet',      # 'snippet' includes title, description, publishedAt, etc.
            type='video',        # We only want videos, not channels or playlists
            order='date',        # Order by date to get the most recent ones first
            maxResults=DAILY_VIDEOS_LIMIT,       # Get up to 5 results (the max per page)
            publishedAfter=published_after_str,
            videoDuration='medium'
        )
        
        response = request.execute()

        for item in response.get('items', []):
            video_id = item['id']['videoId']
            snippet = item['snippet']
            
            videos.append({
                'id': video_id,
                'title': snippet['title'],
                'published_at': snippet['publishedAt'],
                'description': snippet['description'],
                'channel_title': snippet['channelTitle']
            })
            
        return videos

    except Exception as e:
        # Use your logger here in a real application!
        log.info(f"An error occurred with the YouTube API search: {e}")
        return []

def main():
    try:
        initialize_db()
        log.info("Database initialized (or verified to exist).")
    except Exception as e:
        log.critical(f"FATAL: Could not initialize database. Aborting run. Error: {e}")
        return # Stop execution if the database can't be set up
    
    global llm_tool
    llm_tool = LLMTool(llm_model='gemini-pro')

    log.info("--- Starting Daily YouTube Car Search ---")

    recent_videos = search_recent_videos(query="2nd hand cars bangalore", days=1)

    if not recent_videos:
        log.info("No new videos found in the last day.")
    else:
        log.info(f"Found {len(recent_videos)} new videos to process.")
    
    processed_videos = 0
    for video in recent_videos:
        if is_video_processed(video['id']):
            log.info(f"Skipping already processed video: {video['id']}")
            continue
        
        log.info(f"Processing video '{video['title']}' ID: {video['id']} published on {video['published_at']}")
        # Run the full scrape pipeline for this video
        df = run_full_scrape_pipeline_for_video(video)
        
        if df is not None and not df.empty:
            add_cars_to_db(df.to_dict(orient='records'))
            log.info(f"Added {len(df)} cars from video {video['id']} to the database.")
            processed_videos += 1
            if processed_videos >= DAILY_VIDEOS_LIMIT:
                log.info(f"Reached daily limit of {DAILY_VIDEOS_LIMIT} videos processed.")
                break # Stop after processing the daily limit
        else:
            log.warning(f"No data was scraped for video ID {video['id']}")
    
    log.info("--- Daily YouTube Car Search Completed ---")

if __name__ == "__main__":
    main()