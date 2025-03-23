import streamlit as st
import cv2
import numpy as np
import os
from io import BytesIO
import json
import base64
import requests

# Page configuration
st.set_page_config(
    page_title="Fruit Ripeness Detector",
    page_icon="🍎",
    layout="wide"
)

# App title and description
st.title("🍎 Fruit Ripeness Detector")
st.markdown("""Select a fruit, then take up to 5 pictures to analyze its ripeness levels""")

# Fruit configuration dictionary
FRUIT_CONFIG = {
    "banana": {
        "emoji": "🍌",
        "name": "Banana",
        "prompt": """
            Analyze this image and identify all bananas.
            Count them and classify each one into one of these categories:
            - Unripe (green)
            - Ripe (yellow with small brown spots)
            - Overripe (significant brown spots or black)
            
            Return the results in this JSON format:
            {
              "total_count": <number>,
              "unripe_count": <number>,
              "ripe_count": <number>,
              "overripe_count": <number>,
              "detailed_analysis": "<brief description of what you see>"
            }
        """
    },
    "apple": {
        "emoji": "🍏",
        "name": "Apple",
        "prompt": """
            Analyze this image and identify all apples.
            Count them and classify each one into one of these categories:
            - Unripe (too firm, not fully colored)
            - Ripe (firm but gives slightly to pressure, fully colored)
            - Overripe (soft, wrinkled skin or bruised)
            
            Return the results in this JSON format:
            {
              "total_count": <number>,
              "unripe_count": <number>,
              "ripe_count": <number>,
              "overripe_count": <number>,
              "detailed_analysis": "<brief description of what you see>"
            }
        """
    },
    "carrot": {
        "emoji": "🥕",
        "name": "Carrot",
        "prompt": """
            Analyze this image and identify all carrots.
            Count them and classify each one into one of these categories:
            - Unripe (too small, pale orange)
            - Ripe (vibrant orange, firm)
            - Overripe (soft, wrinkled, discolored)
            
            Return the results in this JSON format:
            {
              "total_count": <number>,
              "unripe_count": <number>,
              "ripe_count": <number>,
              "overripe_count": <number>,
              "detailed_analysis": "<brief description of what you see>"
            }
        """
    },
    "tomato": {
        "emoji": "🍅",
        "name": "Tomato",
        "prompt": """
            Analyze this image and identify all tomatoes.
            Count them and classify each one into one of these categories:
            - Unripe (green to light red)
            - Ripe (bright red, firm but gives slightly to pressure)
            - Overripe (very soft, dark red or with mold)
            
            Return the results in this JSON format:
            {
              "total_count": <number>,
              "unripe_count": <number>,
              "ripe_count": <number>,
              "overripe_count": <number>,
              "detailed_analysis": "<brief description of what you see>"
            }
        """
    }
}

def capture_image(key):
    """Capture image from webcam with a unique key"""
    # Check if we already have the maximum number of images
    if 'captured_images' in st.session_state and len(st.session_state.captured_images) >= 5:
        st.warning("Maximum 5 images allowed. Please process current images or remove some.")
        return None, None
    
    # Store the current count of images to detect new captures
    previous_count = 0
    if 'captured_images' in st.session_state:
        previous_count = len(st.session_state.captured_images)
        
    img_file_buffer = st.camera_input(f"Take picture", key=f"camera_{key}")
    
    if img_file_buffer is not None:
        # Convert to OpenCV format
        bytes_data = img_file_buffer.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Add button to confirm this image
        if st.button("Add this image to collection", key=f"add_image_{key}"):
            if 'captured_images' in st.session_state and len(st.session_state.captured_images) < 5:
                st.session_state.captured_images.append(img)
                st.session_state.captured_image_files.append(img_file_buffer)
                st.success(f"Image added! ({len(st.session_state.captured_images)}/5)")
                st.rerun()
        
        return img, img_file_buffer
    return None, None

def encode_image_to_base64(image_file):
    """Encode image to base64 string from file buffer"""
    try:
        if isinstance(image_file, BytesIO):
            return base64.b64encode(image_file.getvalue()).decode('utf-8')
        return base64.b64encode(image_file).decode('utf-8')
    except Exception as e:
        st.error(f"Error encoding image: {str(e)}")
        return None

def get_secret(key, default=None):
    """Get a value from streamlit secrets with a default fallback"""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

def analyze_image_with_vision_api(image_file, fruit_type):
    """Analyze image using Vision API from secrets configuration"""
    # Encode image to base64
    base64_image = encode_image_to_base64(image_file)
    if not base64_image:
        return None
    
    # Get settings from secrets.toml
    endpoint = get_secret("azure_endpoint")
    api_key = get_secret("azure_api_key")
    model = get_secret("azure_model")
    api_version = get_secret("api_version", "2024-02-15-preview")
    
    # Check if required settings are available
    if not endpoint or not api_key:
        st.error("Missing API configuration. Please check your secrets.toml file.")
        return None
    
    # Ensure endpoint has the right format
    if not endpoint.startswith(('http://', 'https://')):
        endpoint = 'https://' + endpoint
    endpoint = endpoint.rstrip('/')
    
    # Construct API URL
    api_url = f"{endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}"
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    # Get fruit-specific prompt
    fruit_prompt = FRUIT_CONFIG.get(fruit_type, FRUIT_CONFIG["banana"])["prompt"]
    
    # Prepare payload
    payload = {
        "messages": [
            {"role": "system", "content": f"You are a computer vision assistant specialized in detecting {FRUIT_CONFIG[fruit_type]['name']}s and assessing their ripeness."},
            {"role": "user", "content": [
                {"type": "text", "text": fruit_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        "temperature": 0.7,
        "max_tokens": 800
    }
    
    try:
        # Make the API call
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        
        # Check for HTTP errors
        if response.status_code != 200:
            st.error(f"API Error: {response.status_code}")
            with st.expander("Response Details"):
                st.text(response.text)
            return None
        
        # Parse and return the response
        result = response.json()
        
        # Extract the content from the response
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0].get("message", {}).get("content", "{}")
            
            # Try to parse the content as JSON
            try:
                parsed_result = json.loads(content)
                return parsed_result
            except json.JSONDecodeError as e:
                st.error(f"JSON parse error: {str(e)}")
                st.write("The response is not valid JSON. Here's the content received:")
                st.code(content)
                
                # Try to extract a JSON object from the content if it contains one
                # Often the API might return text with a JSON object embedded
                try:
                    # Look for a pattern that might be JSON (between curly braces)
                    import re
                    json_pattern = r'\{.*\}'
                    matches = re.search(json_pattern, content, re.DOTALL)
                    if matches:
                        possible_json = matches.group(0)
                        parsed_result = json.loads(possible_json)
                        st.success("Found and extracted a JSON object from the response!")
                        return parsed_result
                except:
                    pass
                    
                # If all else fails, create a dummy result based on the content
                st.warning("Creating a fallback response based on the text")
                return {
                    "total_count": 0,
                    "unripe_count": 0,
                    "ripe_count": 0,
                    "overripe_count": 0,
                    "detailed_analysis": "Unable to extract proper JSON. Raw response: " + content[:200] + "..."
                }
        else:
            st.error("Unexpected API response format")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error in API request: {str(e)}")
        return None

def main():
    # Initialize session state for fruit selection if it doesn't exist
    if 'fruit_selected' not in st.session_state:
        st.session_state.fruit_selected = None
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
        
    # Initialize captured images list
    if 'captured_images' not in st.session_state:
        st.session_state.captured_images = []
        st.session_state.captured_image_files = []
    
    # If fruit is not selected, show fruit selection tiles
    if st.session_state.fruit_selected is None:
        st.subheader("Select a fruit to analyze")
        
        # Create a grid of fruit selection tiles
        cols = st.columns(len(FRUIT_CONFIG))
        
        for i, (fruit_id, fruit_data) in enumerate(FRUIT_CONFIG.items()):
            with cols[i]:
                # Display fruit name as text
                st.write(f"**{fruit_data['name']}**", unsafe_allow_html=True)
                
                # Create a clickable rounded button with just emoji
                st.markdown(f"""
                <div style="padding: 15px; text-align: center; border: 1px solid #ddd; 
                     border-radius: 50px; cursor: pointer; width: 80px; height: 80px; 
                     display: flex; align-items: center; justify-content: center;
                     margin: 0 auto; background-color: #f8f9fa;" onclick="
                    document.querySelector('#select_{fruit_id}').click()
                ">
                    <div style="font-size: 40px;">{fruit_data['emoji']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Hidden button to handle the click
                if st.button(f"Select {fruit_data['name']}", key=f"select_{fruit_id}", help=f"Analyze {fruit_data['name']}"):
                    st.session_state.fruit_selected = fruit_id
                    st.rerun()
    
    # If fruit is selected, show the camera interface
    else:
        selected_fruit = st.session_state.fruit_selected
        fruit_data = FRUIT_CONFIG[selected_fruit]
        
        # Show fruit selection and back button
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("← Back to fruits"):
                st.session_state.fruit_selected = None
                st.session_state.analysis_results = []
                st.session_state.captured_images = []
                st.session_state.captured_image_files = []
                st.rerun()
        
        with col2:
            st.subheader(f"Analyzing {fruit_data['emoji']} {fruit_data['name']}")
        
        # Set up to handle 5 images
        images = []
        image_files = []
        
        # Initialize list in session state if not existing
        if 'captured_images' not in st.session_state:
            st.session_state.captured_images = []
            st.session_state.captured_image_files = []
        
        # Container for the camera input
        st.markdown(f"📷 Take pictures of {fruit_data['name']}s (maximum 5)")
        
        # Single camera element
        img, img_file = capture_image("main")
        
        # Display currently captured images
        if len(st.session_state.captured_images) > 0:
            st.subheader("Captured Images")
            
            # Display images horizontally in a single row
            st.subheader("Captured Images")
            
            # Create a single row for all images
            image_cols = st.columns(len(st.session_state.captured_images))
            
            # Display each image
            for idx, img in enumerate(st.session_state.captured_images):
                with image_cols[idx]:
                    st.image(img, channels="BGR", use_column_width=True)
                    if st.button("Remove", key=f"remove_{idx}"):
                        st.session_state.captured_images.pop(idx)
                        st.session_state.captured_image_files.pop(idx)
                        st.rerun()
            
            # Clear all button in a separate row
            if st.button("Clear All Images"):
                st.session_state.captured_images = []
                st.session_state.captured_image_files = []
                st.rerun()
                
        # Use the session state images
        images = st.session_state.captured_images
        image_files = st.session_state.captured_image_files
        
        # Process button
        col1, col2 = st.columns([5, 1])
        with col1:
            process_button = st.button("Process Images", disabled=len(images) == 0)
        
        # Process the images if any are available and the button is clicked
        if len(images) > 0 and process_button:
            st.session_state.analysis_results = []
            
            st.subheader("Analysis Results")
            
            # Process each image
            for idx, (img, img_file) in enumerate(zip(images, image_files)):
                st.subheader(f"Image #{idx+1} Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(img, channels="BGR", use_column_width=True)
                
                with col2:
                    with st.spinner(f"Analyzing image #{idx+1}..."):
                            # Process the image using the API
                            results = analyze_image_with_vision_api(img_file, selected_fruit)
                            
                            if results:
                                # Store results
                                st.session_state.analysis_results.append(results)
                                
                                # Display results
                                col_total, col_unripe, col_ripe, col_overripe = st.columns(4)
                                col_total.metric("Total Count", results.get("total_count", 0))
                                col_unripe.metric("Unripe", results.get("unripe_count", 0))
                                col_ripe.metric("Ripe", results.get("ripe_count", 0))
                                col_overripe.metric("Overripe", results.get("overripe_count", 0))
                                
                                st.success("Analysis complete!")
                                st.subheader("Detailed Analysis")
                                st.write(results.get("detailed_analysis", "No detailed analysis available."))
                            else:
                                st.error("Failed to analyze the image. Please check your configuration and try again.")
            
            # Show a summary of all results if there are multiple images
            if len(st.session_state.analysis_results) > 1:
                st.subheader("Overall Summary")
                
                # Calculate totals
                total_count = sum(result.get("total_count", 0) for result in st.session_state.analysis_results)
                unripe_count = sum(result.get("unripe_count", 0) for result in st.session_state.analysis_results)
                ripe_count = sum(result.get("ripe_count", 0) for result in st.session_state.analysis_results)
                overripe_count = sum(result.get("overripe_count", 0) for result in st.session_state.analysis_results)
                
                # Display overall metrics
                col_total, col_unripe, col_ripe, col_overripe = st.columns(4)
                col_total.metric("Total Count", total_count)
                col_unripe.metric("Total Unripe", unripe_count)
                col_ripe.metric("Total Ripe", ripe_count)
                col_overripe.metric("Total Overripe", overripe_count)
                
                # Calculate percentages if there are any fruits detected
                if total_count > 0:
                    st.subheader("Ripeness Distribution")
                    
                    # Calculate percentages
                    unripe_pct = (unripe_count / total_count) * 100
                    ripe_pct = (ripe_count / total_count) * 100
                    overripe_pct = (overripe_count / total_count) * 100
                    
                    # Display as a horizontal bar
                    st.markdown(f"""
                    <div style="width:100%; height:30px; background-color:#f0f0f0; border-radius:5px; display:flex;">
                        <div style="width:{unripe_pct}%; height:100%; background-color:#9CCC65; display:flex; align-items:center; justify-content:center; color:white; font-weight:bold;">
                            {unripe_pct:.1f}%
                        </div>
                        <div style="width:{ripe_pct}%; height:100%; background-color:#FFB74D; display:flex; align-items:center; justify-content:center; color:white; font-weight:bold;">
                            {ripe_pct:.1f}%
                        </div>
                        <div style="width:{overripe_pct}%; height:100%; background-color:#EF5350; display:flex; align-items:center; justify-content:center; color:white; font-weight:bold;">
                            {overripe_pct:.1f}%
                        </div>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-top:5px;">
                        <div>Unripe</div>
                        <div>Ripe</div>
                        <div>Overripe</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Submit button (currently does nothing)
            if len(st.session_state.analysis_results) > 0:
                if st.button("Submit Results"):
                    st.success("Results submitted successfully! (This is a placeholder - no action is taken)")

if __name__ == "__main__":
    main()
