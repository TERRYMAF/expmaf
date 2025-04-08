import streamlit as st
import cv2
import numpy as np
from io import BytesIO
import json
import base64
import requests
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Expiry Date Detector",
    page_icon="📅",
    layout="wide"
)

# App title and description
st.title("📅 Expiry Date Detector")
st.markdown("Take a picture of a product expiry date to detect and analyze it")

# Initialize session state
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'current_image_file' not in st.session_state:
    st.session_state.current_image_file = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'image_processed' not in st.session_state:
    st.session_state.image_processed = False
if 'api_error' not in st.session_state:
    st.session_state.api_error = None
if 'raw_api_response' not in st.session_state:
    st.session_state.raw_api_response = None
if 'api_request_info' not in st.session_state:
    st.session_state.api_request_info = None

def get_prompt():
    """Get the prompt for expiry date detection"""
    prompt = """
        Analyze this image and identify any expiry dates on the product.
        
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
        
        Return the results in this JSON format:
        {{
          "dates_found": <number>,
          "expiry_dates": [
            {{
              "date_text": "<exactly as shown on packaging>",
              "date_type": "<expiry or production>",
              "standardized_date": "<YYYY-MM-DD format>",
              "days_until_expiry": <number of days from current date>,
              "expired": <boolean: true if expired, false if not>
            }}
          ],
          "detailed_analysis": "<brief description of what you see, including package type>"
        }}
    """
    return prompt

def capture_image():
    """Capture an image"""
    # Camera troubleshooting expander
    with st.expander("Camera not working? Click here for help"):
        st.markdown("""
        ### Camera Troubleshooting
        
        If you cannot see the camera:
        
        1. **Check browser permissions** - You may need to allow camera access in your browser
        2. **Try a different browser** - Chrome works best with Streamlit
        3. **Make sure your camera is connected** - Check if your webcam is working in other applications
        4. **Refresh the page** - Sometimes a simple refresh can fix camera issues
        """)
    
    # Camera capture with instructions
    st.info("📸 Position the expiry date clearly in the frame. Make sure there's good lighting.")
    img_file_buffer = st.camera_input("Take a picture")
    
    # Process camera image if available
    if img_file_buffer is not None:
        try:
            # Convert to OpenCV format
            bytes_data = img_file_buffer.getvalue()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Store in session state
            st.session_state.current_image = img
            st.session_state.current_image_file = img_file_buffer
            st.session_state.image_processed = False
            st.session_state.analysis_result = None
            st.session_state.api_error = None
            st.rerun()
        except Exception as e:
            st.error(f"Error processing camera image: {str(e)}")

def display_current_image():
    """Display the currently captured image"""
    if st.session_state.current_image is None:
        return
    
    st.subheader("Current Image")
    
    # Create a container with border for the image
    st.markdown("""
    <style>
    .image-container {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        background-color: #f8f9fa;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image(st.session_state.current_image, channels="BGR", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Delete image button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Take a Different Picture"):
            st.session_state.current_image = None
            st.session_state.current_image_file = None
            st.session_state.image_processed = False
            st.session_state.analysis_result = None
            st.session_state.api_error = None
            st.rerun()

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

def analyze_image(image_file):
    """Analyze image using Vision API"""
    # Clear previous errors
    st.session_state.api_error = None
    st.session_state.raw_api_response = None
    
    # Check if the file is too large (API limit is typically 4MB)
    if hasattr(image_file, 'size') and image_file.size > 4 * 1024 * 1024:
        st.session_state.api_error = "Image file is too large (>4MB). Please try a smaller image."
        return None
    
    # Encode image
    base64_image = encode_image_to_base64(image_file)
    if not base64_image:
        st.session_state.api_error = "Failed to encode image"
        return None
    
    # Get API configuration from secrets
    endpoint = get_secret("azure_endpoint")
    api_key = get_secret("azure_api_key")
    model = get_secret("azure_model")
    
    # Validate API configuration
    missing_configs = []
    if not endpoint:
        missing_configs.append("Azure Endpoint")
    if not api_key:
        missing_configs.append("API Key")
    if not model:
        missing_configs.append("Model")
        
    if missing_configs:
        error_msg = f"Missing API configuration: {', '.join(missing_configs)}"
        st.session_state.api_error = error_msg
        return None
    
    # Ensure endpoint has the right format
    if not endpoint.startswith(('http://', 'https://')):
        endpoint = 'https://' + endpoint
    endpoint = endpoint.rstrip('/')
    
    # Construct API URL
    api_url = f"{endpoint}/openai/deployments/{model}/chat/completions?api-version=2024-02-15-preview"
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    # Get prompt
    prompt = get_prompt()
    
    # Prepare payload
    payload = {
        "messages": [
            {"role": "system", "content": "You are a computer vision assistant specialized in detecting expiry dates on products."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        "temperature": 0.5,
        "max_tokens": 800
    }
    
    try:
        # Log request information for troubleshooting
        st.session_state.api_request_info = {
            "url": api_url,
            "model": model,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Make the API call with a timeout
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        
        # Check for HTTP errors
        if response.status_code != 200:
            error_msg = f"API Error (Status {response.status_code}): "
            if response.text:
                try:
                    error_details = response.json()
                    if "error" in error_details and "message" in error_details["error"]:
                        error_msg += error_details["error"]["message"]
                    else:
                        error_msg += response.text[:200]
                except:
                    error_msg += response.text[:200]
            st.session_state.api_error = error_msg
            return None
        
        # Parse the response
        result = response.json()
        
        # Extract the content from the response
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0].get("message", {}).get("content", "{}")
            
            # Try to parse the content as JSON
            try:
                # First, check if the content is empty or whitespace only
                if not content or content.isspace():
                    st.session_state.api_error = "API returned an empty response. Please check your API configuration and credits."
                    return None
                
                # Log the raw content for debugging
                st.session_state.raw_api_response = content[:1000]  # Store first 1000 chars for debugging
                
                try:
                    parsed_result = json.loads(content)
                    return parsed_result
                except json.JSONDecodeError as e:
                    # Try to extract JSON if it's embedded in other text
                    import re
                    # This pattern looks for the outermost JSON object
                    json_pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{(?:[^{}])*\}))*\}))*\})'
                    json_matches = re.search(json_pattern, content, re.DOTALL)
                    
                    if json_matches:
                        try:
                            extracted_json = json_matches.group(1)
                            parsed_result = json.loads(extracted_json)
                            return parsed_result
                        except Exception as json_ex:
                            st.session_state.api_error += f"\nJSON extraction failed: {str(json_ex)}"
                    
                    st.session_state.api_error = f"Could not parse API response as JSON: {str(e)}"
                    return None
            except Exception as e:
                st.session_state.api_error = f"Error processing API response: {str(e)}"
                return None
        
        st.session_state.api_error = "Unexpected API response format"
        return None
        
    except requests.exceptions.Timeout:
        st.session_state.api_error = "API request timed out. Please try again."
        return None
    except requests.exceptions.RequestException as e:
        st.session_state.api_error = f"API request error: {str(e)}"
        return None
    except Exception as e:
        st.session_state.api_error = f"Unexpected error: {str(e)}"
        return None

def process_image():
    """Process the current image"""
    if st.session_state.current_image is None or st.session_state.current_image_file is None:
        st.error("No image available to process")
        return
    
    with st.spinner("Analyzing image..."):
        # Process the image with the vision API
        result = analyze_image(st.session_state.current_image_file)
        
        if result:
            st.session_state.analysis_result = result
            st.session_state.image_processed = True
        else:
            st.session_state.image_processed = False
            # Error message will be displayed from the api_error session state

def display_date_card(date_info):
    """Display a single date card with appropriate styling"""
    original_date = date_info.get("date_text", "Unknown")
    date_type = date_info.get("date_type", "Unknown").upper()
    standardized_date = date_info.get("standardized_date", "Unknown")
    days_until_expiry = date_info.get("days_until_expiry", "Unknown")
    is_expired = date_info.get("expired", False)
    
    # Determine background color and status text based on type and expiry
    if date_type.lower() == "production":
        bg_color = "#f0f0f0"  # Light gray for production dates
        status_text = "PRODUCTION DATE"
    else:
        if is_expired:
            bg_color = "#ffcccc"  # Light red for expired dates
            status_text = "EXPIRED"
        else:
            bg_color = "#d9f2d9"  # Light green for unexpired dates
            status_text = "VALID UNTIL"
    
    # Display date card
    st.markdown(f"""
    <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between;">
            <span style="font-weight: bold;">{date_type}</span>
            <span style="font-weight: bold;">{status_text}</span>
        </div>
        <div style="font-size: 1.2em; margin: 5px 0;">{original_date}</div>
        <div>Standardized: {standardized_date}</div>
        {f'<div>Days until expiry: {days_until_expiry}</div>' if date_type.lower() != "production" else ''}
    </div>
    """, unsafe_allow_html=True)

def display_results():
    """Display the analysis results"""
    st.header("📊 Analysis Results")
    
    results = st.session_state.analysis_result
    if not results:
        st.error("No results available")
        return
    
    # Show image and results side by side
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Image")
        st.image(st.session_state.current_image, channels="BGR", use_container_width=True)
    
    with col2:
        st.subheader("Detected Dates")
        if results.get("dates_found", 0) > 0 and "expiry_dates" in results:
            # Display each date card
            for date_info in results["expiry_dates"]:
                display_date_card(date_info)
        else:
            st.warning("No dates detected in this image.")
    
    # Show detailed analysis
    if "detailed_analysis" in results:
        st.subheader("Analysis Details")
        st.write(results["detailed_analysis"])
    
    # Action button
    if st.button("Scan Another Product", type="primary", use_container_width=True):
        st.session_state.current_image = None
        st.session_state.current_image_file = None
        st.session_state.image_processed = False
        st.session_state.analysis_result = None
        st.session_state.api_error = None
        st.rerun()

# Main app flow
def main():
    # Add less intrusive debug section (expandable)
    with st.expander("Troubleshooting Information"):
        st.write("App Status:")
        st.write(f"- Image captured: {'Yes' if st.session_state.current_image is not None else 'No'}")
        st.write(f"- Image processed: {'Yes' if st.session_state.image_processed else 'No'}")
        st.write(f"- Analysis results: {'Available' if st.session_state.analysis_result else 'None'}")
        
        # API configuration check
        endpoint = get_secret("azure_endpoint")
        api_key = get_secret("azure_api_key")
        model = get_secret("azure_model")
        
        st.write("API Configuration:")
        st.write(f"- Endpoint configured: {'Yes' if endpoint else 'No'}")
        st.write(f"- API Key configured: {'Yes' if api_key else 'No'}")
        st.write(f"- Model configured: {'Yes' if model else 'No'}")
        
        if st.button("Reset App"):
            st.session_state.current_image = None
            st.session_state.current_image_file = None
            st.session_state.image_processed = False
            st.session_state.analysis_result = None
            st.session_state.api_error = None
            st.rerun()
    
    # Display API errors if any
    if st.session_state.api_error:
        st.error(f"API Error: {st.session_state.api_error}")
    
    # If we have results, show them
    if st.session_state.image_processed and st.session_state.analysis_result:
        display_results()
    # Otherwise show image capture interface
    else:
        # If we already have an image, show it and the analyze button
        if st.session_state.current_image is not None:
            display_current_image()
            
            # Analyze button
            if st.button("Analyze Expiry Date", type="primary", use_container_width=True):
                process_image()
                st.rerun()
        # Otherwise show the camera capture
        else:
            capture_image()

if __name__ == "__main__":
    main()
