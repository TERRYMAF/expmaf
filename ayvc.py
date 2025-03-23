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
            
            IMPORTANT: 
            1. Differentiate between expiry dates and production/manufacturing dates.
            2. Expiry dates typically come AFTER production dates and are often preceded by symbols like:
               - "E" or "EXP" or "Expiry" or "Best Before" or "Use By" or "BB"
               - Sometimes format: "E: 01/2025" or "EXP: 01/2025"
            3. Production dates may be labeled with:
               - "P" or "MFG" or "Production Date" or "Produced On"
               - Sometimes format: "P: 01/2024" or "MFG: 01/2024"
            4. When multiple dates are present, the LATER date is most likely the expiry date.
            5. If only month and year are provided (MM/YYYY), assume day 1 of that month.
            
            Return the results in this JSON format without showing the raw JSON to the user:
            {
              "dates_found": <number>,
              "expiry_dates": [
                {
                  "date_text": "<exactly as shown on packaging>",
                  "date_type": "<expiry or production>",
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
            
            IMPORTANT: 
            1. Differentiate between expiry dates and production/manufacturing dates.
            2. Expiry dates typically come AFTER production dates and are often preceded by symbols like:
               - "E" or "EXP" or "Expiry" or "Best Before" or "Use By" or "BB"
               - Sometimes format: "E: 01/2025" or "EXP: 01/2025"
            3. Production dates may be labeled with:
               - "P" or "MFG" or "Production Date" or "Produced On"
               - Sometimes format: "P: 01/2024" or "MFG: 01/2024"
            4. When multiple dates are present, the LATER date is most likely the expiry date.
            5. Dairy products often have shorter shelf lives, so dates that are closer to the current date are likely expiry dates.
            6. If only month and year are provided (MM/YYYY), assume day 1 of that month.
            
            Return the results in this JSON format without showing the raw JSON to the user:
            {
              "dates_found": <number>,
              "expiry_dates": [
                {
                  "date_text": "<exactly as shown on packaging>",
                  "date_type": "<expiry or production>",
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
            Analyze this image and identify any expiry dates on medicine packaging or bottles.
            
            IMPORTANT: 
            1. Differentiate between expiry dates and manufacturing/production dates.
            2. Expiry dates typically come AFTER production dates and are often preceded by symbols like:
               - "E" or "EXP" or "Expiry" or "Use Before"
               - Sometimes format: "E: 01/2025" or "EXP: 01/2025"
            3. Manufacturing dates may be labeled with:
               - "P" or "MFG" or "Manufactured On"
               - Sometimes format: "P: 01/2024" or "MFG: 01/2024"
            4. When multiple dates are present, the LATER date is most likely the expiry date.
            5. Medicine usually has longer shelf lives, so dates that are years in the future are likely expiry dates.
            6. If only month and year are provided (MM/YYYY), assume day 1 of that month.
            
            Return the results in this JSON format without showing the raw JSON to the user:
            {
              "dates_found": <number>,
              "expiry_dates": [
                {
                  "date_text": "<exactly as shown on packaging>",
                  "date_type": "<expiry or production>",
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
            
            IMPORTANT: 
            1. Differentiate between expiry dates and manufacturing/production dates.
            2. Expiry dates typically come AFTER production dates and are often preceded by symbols like:
               - "E" or "EXP" or "Expiry"
               - Sometimes format: "E: 01/2025" or "EXP: 01/2025"
            3. Manufacturing dates may be labeled with:
               - "P" or "MFG" or "Manufacturing Date"
               - Sometimes format: "P: 01/2024" or "MFG: 01/2024"
            4. When multiple dates are present, the LATER date is most likely the expiry date.
            5. Look for a symbol that looks like an open jar with a number and "M" (e.g., "12M") - this is the Period After Opening symbol.
            6. Batch codes are NOT expiry dates (they usually have letters mixed with numbers).
            7. If only month and year are provided (MM/YYYY), assume day 1 of that month.
            
            Return the results in this JSON format without showing the raw JSON to the user:
            {
              "dates_found": <number>,
              "expiry_dates": [
                {
                  "date_text": "<exactly as shown on packaging>",
                  "date_type": "<expiry or production>",
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
    
    # Camera capture with instructions
    st.info("📸 Position the expiry date clearly in the frame. Make sure there's good lighting.")
    img_file_buffer = st.camera_input(f"Take a picture", key=f"camera_{key}")
    
    if img_file_buffer is not None:
        # Convert to OpenCV format
        bytes_data = img_file_buffer.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Process the image immediately
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
        with st.spinner("Analyzing image..."):
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
                    # Look for JSON content in the response
                    import re
                    json_pattern = r'\{[\s\S]*\}'
                    matches = re.search(json_pattern, content)
                    if matches:
                        json_str = matches.group(0)
                        parsed_result = json.loads(json_str)
                    else:
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
                                    # Skip problematic dates rather than showing errors
                                    date_info["days_until_expiry"] = 0
                                    date_info["expired"] = False
                    
                    return parsed_result
                except json.JSONDecodeError:
                    # Create a minimal valid result object
                    return {
                        "dates_found": 0,
                        "expiry_dates": [],
                        "detailed_analysis": "Could not detect any dates in this image."
                    }
            else:
                return {
                    "dates_found": 0,
                    "expiry_dates": [],
                    "detailed_analysis": "Analysis failed to produce results."
                }
                
    except requests.exceptions.RequestException:
        return {
            "dates_found": 0,
            "expiry_dates": [],
            "detailed_analysis": "Network error during analysis."
        }

def process_images():
    """Process the captured images - moved to a function for cleaner code"""
    st.session_state.analysis_results = []
    
    # Process each image
    for idx, (img, img_file) in enumerate(zip(st.session_state.captured_images, st.session_state.captured_image_files)):
        # Process the image using the API
        results = analyze_image_with_vision_api(img_file, st.session_state.product_selected)
        
        if results:
            # Store results
            st.session_state.analysis_results.append(results)
    
    # Set a flag to show we've processed the images
    st.session_state.images_processed = True

def display_results():
    """Display the analysis results in a user-friendly format"""
    st.header("📊 Analysis Results")
    
    # Process each image result
    for idx, results in enumerate(st.session_state.analysis_results):
        with st.expander(f"Image #{idx+1} Results", expanded=True):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(st.session_state.captured_images[idx], channels="BGR", use_column_width=True)
            
            with col2:
                if results.get("dates_found", 0) > 0 and "expiry_dates" in results:
                    # Create a card-like container for each date
                    for date_idx, date_info in enumerate(results["expiry_dates"]):
                        # Extract date information
                        original_date = date_info.get("date_text", "Unknown")
                        date_type = date_info.get("date_type", "Unknown").upper()
                        standardized_date = date_info.get("standardized_date", "Unknown")
                        days_until = date_info.get("days_until_expiry", "Unknown")
                        expired = date_info.get("expired", False)
                        
                        # Determine background color based on status
                        if date_type.lower() == "production":
                            bg_color = "#f0f0f0"  # Light gray for production dates
                            status_text = "PRODUCTION DATE"
                        elif expired:
                            bg_color = "#ffcccc"  # Light red for expired
                            status_text = "EXPIRED"
                        elif days_until <= 7:
                            bg_color = "#fff2cc"  # Light yellow for expiring soon
                            status_text = "EXPIRING SOON"
                        else:
                            bg_color = "#d9f2d9"  # Light green for valid
                            status_text = "VALID"
                        
                        # Display date card
                        st.markdown(f"""
                        <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="font-weight: bold;">{date_type}</span>
                                <span style="font-weight: bold;">{status_text}</span>
                            </div>
                            <div style="font-size: 1.2em; margin: 5px 0;">{original_date}</div>
                            <div>Standardized: {standardized_date}</div>
                        """, unsafe_allow_html=True)
                        
                        # Add days until expiry info if it's an expiry date
                        if date_type.lower() == "expiry":
                            if expired:
                                st.markdown(f"""
                                <div style="color: red;">Expired {abs(days_until)} days ago</div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div>Days until expiry: {days_until}</div>
                                """, unsafe_allow_html=True)
                        
                        # Close the div
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning("No dates detected in this image.")
                
                # Display PAO information for cosmetics if available
                if st.session_state.product_selected == "cosmetics" and results.get("pao_found", False):
                    pao_months = results.get("pao_months", "Unknown")
                    st.info(f"📌 Period After Opening (PAO): {pao_months} months")
    
    # Show a summary of all results
    if len(st.session_state.analysis_results) > 0:
        st.header("📝 Summary")
        
        # Get all expiry dates
        all_dates = []
        for result in st.session_state.analysis_results:
            if "expiry_dates" in result:
                all_dates.extend(result["expiry_dates"])
        
        # Filter to just expiry dates (not production)
        expiry_only = [date for date in all_dates if date.get("date_type", "").lower() == "expiry"]
        production_only = [date for date in all_dates if date.get("date_type", "").lower() == "production"]
        
        # Count expired, expiring soon, and valid dates
        expired_count = sum(1 for date in expiry_only if date.get("expired", False))
        expiring_soon = sum(1 for date in expiry_only if not date.get("expired", False) and date.get("days_until_expiry", 999) <= 7)
        valid_count = sum(1 for date in expiry_only if not date.get("expired", False) and date.get("days_until_expiry", 0) > 7)
        
        # Create the summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Expired", expired_count, delta=None, delta_color="inverse")
        with col2:
            st.metric("Expiring Soon", expiring_soon, delta=None, delta_color="inverse")
        with col3:
            st.metric("Valid", valid_count, delta=None, delta_color="normal")
        
        # Display oldest and newest expiry dates
        if expiry_only:
            st.subheader("Expiry Date Range")
            
            # Sort by standardized dates
            sorted_dates = sorted(expiry_only, key=lambda x: x.get("standardized_date", "9999-99-99"))
            
            # Get earliest and latest dates
            earliest = sorted_dates[0] if sorted_dates else None
            latest = sorted_dates[-1] if sorted_dates else None
            
            if earliest and latest:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Earliest Expiry Date:**")
                    earliest_date = earliest.get("standardized_date", "Unknown")
                    days_until = earliest.get("days_until_expiry", "Unknown")
                    expired = earliest.get("expired", False)
                    
                    if expired:
                        st.markdown(f"""
                        <div style="background-color: #ffcccc; padding: 10px; border-radius: 5px;">
                            <div style="font-size: 1.2em;">{earliest_date}</div>
                            <div style="color: red;">Expired {abs(days_until)} days ago</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color: {'#fff2cc' if days_until <= 7 else '#d9f2d9'}; padding: 10px; border-radius: 5px;">
                            <div style="font-size: 1.2em;">{earliest_date}</div>
                            <div>Days until expiry: {days_until}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Latest Expiry Date:**")
                    latest_date = latest.get("standardized_date", "Unknown")
                    days_until = latest.get("days_until_expiry", "Unknown")
                    expired = latest.get("expired", False)
                    
                    if expired:
                        st.markdown(f"""
                        <div style="background-color: #ffcccc; padding: 10px; border-radius: 5px;">
                            <div style="font-size: 1.2em;">{latest_date}</div>
                            <div style="color: red;">Expired {abs(days_until)} days ago</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color: {'#fff2cc' if days_until <= 7 else '#d9f2d9'}; padding: 10px; border-radius: 5px;">
                            <div style="font-size: 1.2em;">{latest_date}</div>
                            <div>Days until expiry: {days_until}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Create a simple "Export" or "Save" button that doesn't actually do anything
        if st.button("Save Results", type="primary", use_container_width=True):
            st.success("Results saved! (This is a placeholder - no action is taken)")
            
        # Start over button
        if st.button("Start Over", type="secondary", use_container_width=True):
            st.session_state.product_selected = None
            st.session_state.analysis_results = []
            st.session_state.captured_images = []
            st.session_state.captured_image_files = []
            st.session_state.images_processed = False
            st.rerun()

def main():
    # Set up session state
    if 'product_selected' not in st.session_state:
        st.session_state.product_selected = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'captured_images' not in st.session_state:
        st.session_state.captured_images = []
        st.session_state.captured_image_files = []
    if 'images_processed' not in st.session_state:
        st.session_state.images_processed = False
    
    # If product is not selected, show product selection tiles
    if st.session_state.product_selected is None:
        st.subheader("Select product type")
        
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
                if st.button(f"Select {product_data['name']}", key=f"select_{product_id}", help=f"Analyze {product_data['name']}", label_visibility="collapsed"):
                    st.session_state.product_selected = product_id
                    st.rerun()
    
    # If product is selected, show the camera interface
    else:
        selected_product = st.session_state.product_selected
        product_data = PRODUCT_CONFIG[selected_product]
        
        # Show product selection and back button
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("←", help="Back to product selection"):
                st.session_state.product_selected = None
                st.session_state.analysis_results = []
                st.session_state.captured_images = []
                st.session_state.captured_image_files = []
                st.session_state.images_processed = False
                st.rerun()
        
        with col2:
            st.title(f"{product_data['emoji']} {product_data['name']} Expiry Detector")
        
        # Set up to handle 5 images
        images = []
        image_files = []
        
        # Initialize list in session state if not existing
        if 'captured_images' not in st.session_state:
            st.session_state.captured_images = []
            st.session_state.captured_image_files = []
        
        # Container for the image collection
        st.markdown("📷 Take pictures of expiry dates")
        
        # Simple camera element
        img, img_file = capture_image("main")
        
        # Display currently captured images
        if len(st.session_state.captured_images) > 0:
            # Display images in a simplified view
            st.subheader("Captured Images")
            
            # Create a container with border
            st.markdown("""
            <style>
            .image-gallery {
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 10px;
                background-color: #f8f9fa;
                margin-bottom: 20px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="image-gallery">', unsafe_allow_html=True)
            
            # Create a grid layout for images
            num_images = len(st.session_state.captured_images)
            cols_per_row = min(3, num_images)  # Up to 3 images per row
            
            if num_images > 0:
                # Calculate number of rows needed
                rows = (num_images + cols_per_row - 1) // cols_per_row
                
                for row in range(rows):
                    # Create columns for this row
                    cols = st.columns(cols_per_row)
                    
                    # Fill each column with an image
                    for col in range(cols_per_row):
                        idx = row * cols_per_row + col
                        if idx < num_images:
                            with cols[col]:
                                st.image(st.session_state.captured_images[idx], channels="BGR", use_column_width=True)
                                if st.button("❌", key=f"remove_{idx}", help="Remove this image"):
                                    st.session_state.captured_images.pop(idx)
                                    st.session_state.captured_image_files.pop(idx)
                                    st.rerun()
