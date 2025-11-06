# build_kb.py
import os
import glob
import json
import sqlite3 # Keep sqlite3 import if potentially needed elsewhere, though not used here
import numpy as np
import time
import hashlib
import re
import uuid
import logging
import sys
from typing import List, Dict, Tuple, Any, Optional

# ---
# This script is designed to be run manually to build your Knowledge Base.
# It imports your 'utils.py' file.
#
# ASSUMPTIONS:
# 1. You have a valid 'utils.py' in the same directory.
# 2. 'utils.manual_cohere_embed' will RAISE an Exception on batch failure.
# 3. 'utils.COHERE_API_KEY' is set in your .env file.
# ---
try:
    import utils
except ImportError:
    print("FATAL ERROR: 'utils.py' not found.")
    print("Please ensure this script is in the same directory as 'utils.py'.")
    sys.exit(1)

# --- Configuration ---
# Load env vars to get the KB_GLOB path
from dotenv import load_dotenv
load_dotenv()
# --- ADD DEBUG PRINTS ---
print(f"DEBUG: Current Working Directory = {os.getcwd()}")
print(f"DEBUG: KB_GLOB from .env = {os.getenv('KB_GLOB')}")
# ------------------------
KB_GLOB = os.getenv("KB_GLOB", "./demo-kb/**/*.json") # Keep the default fallback
# --- ADD DEBUG PRINT ---
print(f"DEBUG: Final KB_GLOB pattern being used = {KB_GLOB}")
# -----------------------

# Define our consistent output filenames
OUTPUT_VECTOR_STORE = "vector_store.npy"
OUTPUT_KB_DATA = "kb_data.json"

# Set up logging to print to the console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

# ==============================================================================
# --- KB CHUNKING LOGIC ---
# (This is moved from agentic_backend.py)
# ==============================================================================

def load_and_chunk_knowledge_base(kb_glob: str = KB_GLOB) -> List[Dict[str, Any]]:
    """
    Loads KB files, splits them into logical chunks, and returns
    a single list of data dictionaries.
    """

    # This list will hold all our final chunk dictionaries
    kb_data_items: List[Dict[str, Any]] = []
    # --- ADD DEBUG PRINT ---
    print(f"DEBUG: Inside load_and_chunk_knowledge_base, using pattern: {kb_glob}")
    # -----------------------

    # Determine if recursive is needed based on pattern
    recursive_glob = '**' in kb_glob

    files = glob.glob(kb_glob, recursive=recursive_glob) # Use recursive flag correctly
    # --- ADD DEBUG PRINT ---
    print(f"DEBUG: glob.glob found files: {files}")
    # -----------------------

    if not files:
        logger.warning(f"No files found matching KB_GLOB pattern: {kb_glob}")
        return []

    logger.info(f"Found {len(files)} files to process...")
    file_count = 0 # <-- ADDED COUNTER
    chunk_count = 0 # <-- ADDED COUNTER

    for f in files:
        file_count += 1 # <-- INCREMENT COUNTER
        print(f"DEBUG: Attempting to process file #{file_count}: {f}")
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                # --- ADDED DEBUG PRINT ---
                print(f"DEBUG: Successfully loaded JSON from {f}. Type: {type(data)}")
                # -------------------------
                items = data if isinstance(data, list) else [data]
                # --- ADDED DEBUG PRINT ---
                print(f"DEBUG: Processing {len(items)} item(s) from {f}")
                # -------------------------

                for item_index, item in enumerate(items):
                    # --- ADDED DEBUG PRINT ---
                    print(f"DEBUG: Processing item {item_index + 1}/{len(items)} in {f}")
                    # -------------------------
                    # Extract common metadata
                    article_id = item.get("article_number", f"file-{hashlib.md5(f.encode()).hexdigest()[:8]}")
                    title = item.get("title", "Untitled")
                    category = item.get("category", "General")

                    # --- This is our new, aligned 'add_chunk' function ---
                    # It adds a single, complete dictionary to our list
                    def add_chunk(text: str, section: str):
                        nonlocal chunk_count # <-- Allow modifying outer counter
                        if text and text.strip():
                            kb_data_items.append({
                                "text_chunk": text,  # <-- The text itself
                                "source": article_id,
                                "section": section,
                                "category": category,
                                "title": title
                            })
                            chunk_count += 1 # <-- INCREMENT COUNTER
                            # --- ADDED DEBUG PRINT ---
                            print(f"DEBUG: Added chunk #{chunk_count} (Section: {section}, Size: {len(text)} chars)")
                            # -------------------------
                        # --- ADDED DEBUG PRINT ---
                        # else:
                        #     print(f"DEBUG: Skipped adding chunk (Section: {section}) because text was empty.")
                        # -------------------------

                    # Overview chunk
                    overview = f"Title: {title}. Problem: {item.get('problem_statement', '')}. Summary: {item.get('summary', '')}"
                    add_chunk(overview, "Overview")

                    # Solution chunk
                    solution = f"Title: {title}. Solution: {item.get('solution_answer', '')}"
                    add_chunk(solution, "Solution")

                    # Instructions chunks
                    instructions = item.get("step_by_step_instructions", [])
                    current_section = "General Instructions"
                    current_steps = []

                    for step in instructions:
                        if isinstance(step, str):
                            if step.startswith("###"):
                                if current_steps:
                                    chunk_text = f"Title: {title}. Section: {current_section}. Steps: {' '.join(current_steps)}"
                                    add_chunk(chunk_text, current_section)
                                current_section = step.replace("#", "").strip()
                                current_steps = []
                            else:
                                current_steps.append(step)

                    if current_steps: # Add the last section
                        chunk_text = f"Title: {title}. Section: {current_section}. Steps: {' '.join(current_steps)}"
                        add_chunk(chunk_text, current_section)

                    # Scenario chunks
                    scenarios = item.get("contextual_scenarios", [])
                    for i, scenario in enumerate(scenarios):
                        if isinstance(scenario, dict):
                            scenario_text = f"Title: {title}. Scenario: {scenario.get('user_scenario', '')}. Cause: {scenario.get('likely_cause', '')}. Actions: {' '.join(scenario.get('immediate_actions', []))}"
                            add_chunk(scenario_text, f"Scenario {i+1}")

        except json.JSONDecodeError as json_err:
            logger.error(f"Could not decode JSON from {f}. Error: {json_err}") # <-- More specific error
        except Exception as e:
            logger.error(f"Error processing file {f}: {e}", exc_info=True) # <-- Add exc_info

    # --- ADDED DEBUG PRINT ---
    print(f"DEBUG: Finished processing all files. Total chunks added: {len(kb_data_items)}")
    # -------------------------
    logger.info(f"Loaded and processed {len(kb_data_items)} total chunks from {len(files)} files.")
    return kb_data_items

# ==============================================================================
# --- MAIN BUILD SCRIPT ---
# ==============================================================================

def main():
    """
    Runs the end-to-end KB indexing pipeline.
    1. Loads and chunks data.
    2. Embeds data using Cohere.
    3. Normalizes vectors.
    4. Saves 'vector_store.npy' and 'kb_data.json' to disk.
    """
    logger.info("--- 🚀 Starting Knowledge Base Build Pipeline ---")

    try:
        # --- 1. Load and Chunk Data ---
        logger.info(f"Step 1/4: Loading and chunking data from '{KB_GLOB}'...")
        kb_data_items = load_and_chunk_knowledge_base(KB_GLOB)

        if not kb_data_items:
            logger.error("No data items were loaded. Check JSON structure and KB_GLOB. Aborting pipeline.") # <-- More specific message
            return

        # Extract just the texts for the embedding API
        chunk_texts = [item['text_chunk'] for item in kb_data_items]
        # --- ADDED DEBUG PRINTS ---
        print(f"DEBUG: Extracted {len(chunk_texts)} text chunks for embedding.")
        if chunk_texts:
             print(f"DEBUG: First chunk preview: '{chunk_texts[0][:100]}...'")
        else:
             print("DEBUG: chunk_texts list is empty!")
             return # Exit if no texts
        # -------------------------


        # --- 2. Embed Chunks via Cohere API ---
        logger.info("Step 2/4: Calling Cohere API (utils.manual_cohere_embed)...")
        logger.warning("This may take several minutes and will use your API quota.")

        start_time = time.time()
        # --- ADDED DEBUG PRINT ---
        print(f"DEBUG: Calling utils.manual_cohere_embed with {len(chunk_texts)} texts...")
        # -------------------------
        try:
            chunk_embeddings = utils.manual_cohere_embed(
                texts=chunk_texts,
                input_type="search_document"
            )
        except Exception as embed_err:
             # --- ADDED DEBUG PRINT ---
             print(f"DEBUG: utils.manual_cohere_embed RAISED AN EXCEPTION: {embed_err}")
             # -------------------------
             raise # Re-raise the exception to be caught below

        end_time = time.time()
        # --- ADDED DEBUG PRINTS ---
        print(f"DEBUG: utils.manual_cohere_embed returned. Result type: {type(chunk_embeddings)}")
        if isinstance(chunk_embeddings, list):
            print(f"DEBUG: Received {len(chunk_embeddings)} embeddings.")
        # -------------------------

        if not chunk_embeddings or len(chunk_embeddings) != len(chunk_texts):
            logger.error(f"Embedding failed or returned incomplete results. Expected {len(chunk_texts)}, got {len(chunk_embeddings) if chunk_embeddings else 0}.")
            # --- ADDED DEBUG PRINT ---
            print(f"DEBUG: Exiting because embedding results are invalid.")
            # -------------------------
            return

        logger.info(f"Successfully received {len(chunk_embeddings)} embeddings in {end_time - start_time:.2f} seconds.")

        # --- 3. Process and Normalize Vectors ---
        logger.info("Step 3/4: Normalizing vectors for efficient cosine similarity...")

        vector_store = np.array(chunk_embeddings).astype(np.float32)

        # Normalize vectors (as required by our core_rag.py review)
        norms = np.linalg.norm(vector_store, axis=1, keepdims=True)
        # Add epsilon to prevent division by zero for any zero-vectors
        norms[norms == 0] = 1e-9
        vector_store_normalized = vector_store / norms

        logger.info("Vector normalization complete.")

        # --- 4. Save Artifacts to Disk ---
        logger.info(f"Step 4/4: Saving artifacts to disk...")

        # Save the normalized vectors
        np.save(OUTPUT_VECTOR_STORE, vector_store_normalized)
        logger.info(f"✅ Saved normalized vector store to '{OUTPUT_VECTOR_STORE}'")

        # Save the combined metadata + text file
        with open(OUTPUT_KB_DATA, 'w', encoding='utf-8') as f:
            json.dump(kb_data_items, f, indent=2)
        logger.info(f"✅ Saved Knowledge Base data to '{OUTPUT_KB_DATA}'")

        logger.info("--- 🎉 Knowledge Base Build Complete! ---")
        logger.info("Your application is now ready to load the new KB on its next restart.")

    except Exception as e:
        logger.error(f"--- ❌ PIPELINE FAILED ---")
        logger.error(f"An unexpected error occurred: {e}", exc_info=True) # <-- Add exc_info
        if "Embedding batch failed" in str(e):
             logger.error("The failure occurred during the Cohere embedding step. Check API key/status.")
        sys.exit(1) # Ensure script exits on error

if __name__ == "__main__":
    main()
