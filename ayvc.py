import streamlit as st
import cv2
import numpy as np
import os
from io import BytesIO
import json
import base64
import requests
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Expiry Date Detector",
    page_icon="📅",
    layout="wide"
)

# App title and description
st.title("📅 Expiry Date Detector")
st.markdown("""Take pictures of product expiry dates to detect and track them automatically""")

# Product configuration dictionary
PRODUCT_CONFIG = {
    "packaged_food": {
        "emoji": "🥫",
        "name": "Packaged Food",
        "prompt": """
            Analyze this image and identify any expiry dates, best-before dates, or use-by dates on packaged food products.
            Extract all visible dates and determine their format (DD/MM/YYYY, MM/YYYY, etc.).
            
            If only month and year are provided (MM/YYYY), assume day 1 of that month.
            
            Return the results in this JSON format:
            {
              "dates_found": <number>,
              "expiry_dates": [
                {
                  "date_text": "<exactly as shown on packaging>",
                  "standardized_date": "<YYYY-MM-DD format>",
                  "days_until_expiry": <number of days from current date>,
                  "expired": <boolean: true if expired, false if not>
                }
              ],
              "detailed_analysis": "<brief description of what you see, including package type>"
            }
        """
    },
    "dairy": {
        "emoji": "🥛",
        "name": "Dairy Products",
        "prompt": """
            Analyze this image and identify any expiry dates, best-before dates, or use-by dates on dairy products.
            Extract all visible dates and determine their format (DD/MM/YYYY, MM/YYYY, etc.).
            
            If only month and year are provided (MM/YYYY), assume day 1 of that month.
            
            Return the results in this JSON format:
            {
              "dates_found": <number>,
              "expiry_dates": [
                {
                  "date_text": "<exactly as shown on packaging>",
                  "standardized_date": "<YYYY-MM-DD format>",
                  "days_until_expiry": <number of days from current date>,
                  "expired": <boolean: true if expired, false if not>
                }
              ],
              "detailed_analysis": "<brief description of what you see, including product type>"
            }
        """
    },
    "medicine": {
        "emoji": "💊",
        "name": "Medicine",
        "prompt": """
            Analyze this image and identify any expiry dates or manufacturing dates on medicine packaging or bottles.
            Extract all visible dates and determine their format (DD/MM/YYYY, MM/YYYY, etc.).
            
            If only month and year are provided (MM/YYYY), assume day 1 of that month.
            
            Return the results in this JSON format:
            {
              "dates_found": <number>,
              "expiry_dates": [
                {
                  "date_text": "<exactly as shown on packaging>",
                  "standardized_date": "<YYYY-MM-DD format>",
                  "days_until_expiry": <number of days from current date>,
                  "expired": <boolean: true if expired, false if not>
                }
              ],
              "detailed_analysis": "<brief description of what you see, including medication type if visible>"
            }
        """
    },
    "cosmetics": {
        "emoji": "🧴",
        "name": "Cosmetics",
        "prompt": """
            Analyze this image and identify any expiry dates, Period After Opening (PAO) symbols, or manufacturing dates on cosmetic products.
            Extract all visible dates and determine their format (DD/MM/YYYY, MM/YYYY, etc.).
            Look for PAO symbols (e.g., "12M" meaning 12 months after opening).
            
            If only month and year are provided (MM/YYYY), assume day 1 of that month.
            
            Return the results in this JSON format:
            {
              "dates_found": <number>,
              "expiry_dates": [
                {
                  "date_text": "<exactly as shown on packaging>",
                  "standardized_date": "<YYYY-MM-DD format>",
                  "days_until_expiry": <number of days from current date>,
                  "expired": <boolean: true if expired, false if not>
                }
              ],
              "pao_found": <boolean>,
              "pao_months": <number or null>,
              "detailed_analysis": "<brief description of what you see, including product type>"
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
    
    # File uploader option
    st.write("Option 1: Upload an image from your device")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key=f"uploader_{key}")
    
    if uploaded_file is not None:
        # Convert to OpenCV format
        bytes_data = uploaded_file.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        st.image(img, channels="BGR", caption="Uploaded Image", use_column_width=True)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Add to collection", key=f"add_uploaded_{key}"):
                if 'captured_images' in st.session_state and len(st.session_state.captured_images) < 5:
                    st.session_state.captured_images.append(img)
                    st.session_state.captured_image_files.append(uploaded_file)
                    st.success(f"Image added! ({len(st.session_state.captured_images)}/5)")
                    st.rerun()
        return img, uploaded_file
    
    # Camera option
    st.write("Option 2: Take a picture with your camera")
    img_file_buffer = st.camera_input(f"Take picture", key=f"camera_{key}")
    
    if img_file_buffer is not None:
        # Convert to OpenCV format
        bytes_data = img_file_buffer.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Add a preview and review section
        st.write("Review your image:")
        st.image(img, channels="BGR", caption="Preview", use_column_width=True)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Add to collection", key=f"add_image_{key}"):
                if 'captured_images' in st.session_state and len(st.session_state.captured_images) < 5:
                    st.session_state.captured_images.append(img)
                    st.session_state.captured_image_files.append(img_file_buffer)
                    st.success(f"Image added! ({len(st.session_state.captured_images)}/5)")
                    st.rerun()
        with col2:
            st.write("Take another picture if you're not satisfied with this one.")
        
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

def analyze_image_with_vision_api(image_file, product_type):
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
    
    # Get product-specific prompt
    product_prompt = PRODUCT_CONFIG.get(product_type, PRODUCT_CONFIG["packaged_food"])["prompt"]
    
    # Prepare payload
    payload = {
        "messages": [
            {"role": "system", "content": f"You are a computer vision assistant specialized in detecting expiry dates on {PRODUCT_CONFIG[product_type]['name']}."},
            {"role": "user", "content": [
                {"type": "text", "text": product_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        "temperature": 0.5,
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
                
                # Process the dates - standardize format and calculate days until expiry
                current_date = datetime.now().date()
                
                if "expiry_dates" in parsed_result:
                    for date_info in parsed_result["expiry_dates"]:
                        if "standardized_date" in date_info:
                            try:
                                # Parse the standardized date
                                expiry_date = datetime.strptime(date_info["standardized_date"], "%Y-%m-%d").date()
                                
                                # Calculate days until expiry
                                days_until = (expiry_date - current_date).days
                                date_info["days_until_expiry"] = days_until
                                
                                # Set expired flag
                                date_info["expired"] = days_until < 0
                            except Exception as e:
                                st.warning(f"Could not process date {date_info['standardized_date']}: {str(e)}")
                
                return parsed_result
            except json.JSONDecodeError as e:
                st.error(f"JSON parse error: {str(e)}")
                st.write("The response is not valid JSON. Here's the content received:")
                st.code(content)
                
                # Try to extract a JSON object from the content if it contains one
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
                    "dates_found": 0,
                    "expiry_dates": [],
                    "detailed_analysis": "Unable to extract proper JSON. Raw response: " + content[:200] + "..."
                }
        else:
            st.error("Unexpected API response format")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error in API request: {str(e)}")
        return None

def main():
    # Initialize session state for product selection if it doesn't exist
    if 'product_selected' not in st.session_state:
        st.session_state.product_selected = None
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
        
    # Initialize captured images list
    if 'captured_images' not in st.session_state:
        st.session_state.captured_images = []
        st.session_state.captured_image_files = []
    
    # If product is not selected, show product selection tiles
    if st.session_state.product_selected is None:
        st.subheader("Select a product type to analyze")
        
        # Create a grid of product selection tiles
        cols = st.columns(len(PRODUCT_CONFIG))
        
        for i, (product_id, product_data) in enumerate(PRODUCT_CONFIG.items()):
            with cols[i]:
                # Display product name as text
                st.write(f"**{product_data['name']}**", unsafe_allow_html=True)
                
                # Create a clickable rounded button with just emoji
                st.markdown(f"""
                <div style="padding: 15px; text-align: center; border: 1px solid #ddd; 
                     border-radius: 50px; cursor: pointer; width: 80px; height: 80px; 
                     display: flex; align-items: center; justify-content: center;
                     margin: 0 auto; background-color: #f8f9fa;" onclick="
                    document.querySelector('#select_{product_id}').click()
                ">
                    <div style="font-size: 40px;">{product_data['emoji']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Hidden button to handle the click
                if st.button(f"Select {product_data['name']}", key=f"select_{product_id}", help=f"Analyze {product_data['name']}"):
                    st.session_state.product_selected = product_id
                    st.rerun()
    
    # If product is selected, show the camera interface
    else:
        selected_product = st.session_state.product_selected
        product_data = PRODUCT_CONFIG[selected_product]
        
        # Show product selection and back button
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("← Back to products"):
                st.session_state.product_selected = None
                st.session_state.analysis_results = []
                st.session_state.captured_images = []
                st.session_state.captured_image_files = []
                st.rerun()
        
        with col2:
            st.subheader(f"Analyzing {product_data['emoji']} {product_data['name']} Expiry Dates")
        
        # Set up to handle 5 images
        images = []
        image_files = []
        
        # Initialize list in session state if not existing
        if 'captured_images' not in st.session_state:
            st.session_state.captured_images = []
            st.session_state.captured_image_files = []
        
        # Container for the image collection
        st.markdown(f"📷 Collect images of {product_data['name']} expiry dates (maximum 5)")
        
        # Collection workflow
        st.subheader("Add New Images")
        with st.expander("Click to add a new image", expanded=True):
            img, img_file = capture_image("main")
        
        # Display currently captured images
        if len(st.session_state.captured_images) > 0:
            st.subheader("Captured Images")
            
            # Display images horizontally in a single row
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
                            results = analyze_image_with_vision_api(img_file, selected_product)
                            
                            if results:
                                # Store results
                                st.session_state.analysis_results.append(results)
                                
                                # Display results
                                st.metric("Dates Found", results.get("dates_found", 0))
                                
                                # Display each detected date
                                if results.get("dates_found", 0) > 0 and "expiry_dates" in results:
                                    for date_idx, date_info in enumerate(results["expiry_dates"]):
                                        st.markdown(f"#### Date {date_idx+1}")
                                        
                                        # Create 4 columns for date information
                                        col1, col2, col3, col4 = st.columns(4)
                                        
                                        # Format information
                                        original_date = date_info.get("date_text", "Unknown")
                                        standardized_date = date_info.get("standardized_date", "Unknown")
                                        days_until = date_info.get("days_until_expiry", "Unknown")
                                        expired = date_info.get("expired", False)
                                        
                                        # Display in columns
                                        col1.markdown(f"**Original Text:** {original_date}")
                                        col2.markdown(f"**Standardized:** {standardized_date}")
                                        
                                        # Add color based on days until expiry
                                        if expired:
                                            col3.markdown(f"**Days until expiry:** <span style='color:red;font-weight:bold;'>Expired ({days_until} days)</span>", unsafe_allow_html=True)
                                        elif days_until <= 7:
                                            col3.markdown(f"**Days until expiry:** <span style='color:orange;font-weight:bold;'>{days_until} days</span>", unsafe_allow_html=True)
                                        else:
                                            col3.markdown(f"**Days until expiry:** <span style='color:green;font-weight:bold;'>{days_until} days</span>", unsafe_allow_html=True)
                                        
                                        # Status column
                                        if expired:
                                            col4.error("EXPIRED")
                                        elif days_until <= 7:
                                            col4.warning("EXPIRING SOON")
                                        else:
                                            col4.success("VALID")
                                else:
                                    st.warning("No expiry dates detected in this image.")
                                
                                # Display PAO information for cosmetics if available
                                if selected_product == "cosmetics" and results.get("pao_found", False):
                                    pao_months = results.get("pao_months", "Unknown")
                                    st.info(f"📌 Period After Opening (PAO): {pao_months} months")
                                
                                st.success("Analysis complete!")
                                st.subheader("Detailed Analysis")
                                st.write(results.get("detailed_analysis", "No detailed analysis available."))
                            else:
                                st.error("Failed to analyze the image. Please check your configuration and try again.")
            
            # Show a summary of all results if there are multiple images
            if len(st.session_state.analysis_results) > 1:
                st.subheader("Overall Summary")
                
                # Calculate totals
                total_dates = sum(result.get("dates_found", 0) for result in st.session_state.analysis_results)
                
                # Get all expiry dates
                all_expiry_dates = []
                for result in st.session_state.analysis_results:
                    if "expiry_dates" in result:
                        all_expiry_dates.extend(result["expiry_dates"])
                
                # Count expired, expiring soon, and valid dates
                expired_count = sum(1 for date in all_expiry_dates if date.get("expired", False))
                expiring_soon = sum(1 for date in all_expiry_dates if not date.get("expired", False) and date.get("days_until_expiry", 999) <= 7)
                valid_count = sum(1 for date in all_expiry_dates if not date.get("expired", False) and date.get("days_until_expiry", 0) > 7)
                
                # Display overall metrics
                col_total, col_expired, col_expiring, col_valid = st.columns(4)
                col_total.metric("Total Dates", total_dates)
                col_expired.metric("Expired", expired_count)
                col_expiring.metric("Expiring Soon (≤7 days)", expiring_soon)
                col_valid.metric("Valid", valid_count)
                
                # Show table of all dates if any are found
                if total_dates > 0:
                    st.subheader("All Detected Expiry Dates")
                    
                    # Create a dataframe with all dates
                    import pandas as pd
                    
                    # Prepare data for dataframe
                    data = []
                    for i, result in enumerate(st.session_state.analysis_results):
                        if "expiry_dates" in result:
                            for date in result["expiry_dates"]:
                                status = "EXPIRED" if date.get("expired", False) else "EXPIRING SOON" if date.get("days_until_expiry", 999) <= 7 else "VALID"
                                data.append({
                                    "Image": f"Image #{i+1}",
                                    "Original Text": date.get("date_text", "Unknown"),
                                    "Standardized Date": date.get("standardized_date", "Unknown"),
                                    "Days Until Expiry": date.get("days_until_expiry", "Unknown"),
                                    "Status": status
                                })
                    
                    # Create and display dataframe
                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df, hide_index=True)
            
            # Add "Save to Inventory" button (placeholder for future functionality)
            if len(st.session_state.analysis_results) > 0:
                if st.button("Save to Inventory"):
                    st.success("Expiry dates saved to your inventory! (This is a placeholder - no action is taken)")

if __name__ == "__main__":
    main()
