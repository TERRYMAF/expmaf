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
st.markdown("Take pictures of product expiry dates to detect and track them automatically")

# Product configuration dictionary
PRODUCT_CONFIG = {
    "packaged_food": {
        "emoji": "🥫",
        "name": "Packaged Food"
    },
    "dairy": {
        "emoji": "🥛",
        "name": "Dairy Products"
    },
    "medicine": {
        "emoji": "💊",
        "name": "Medicine"
    },
    "cosmetics": {
        "emoji": "🧴",
        "name": "Cosmetics"
    }
}

# Initialize session state
if 'product_selected' not in st.session_state:
    st.session_state.product_selected = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'captured_images' not in st.session_state:
    st.session_state.captured_images = []
    st.session_state.captured_image_files = []
if 'images_processed' not in st.session_state:
    st.session_state.images_processed = False

def get_prompt(product_type):
    """Get the appropriate prompt for each product type"""
    base_prompt = f"""
        Analyze this image and identify any expiry dates on {PRODUCT_CONFIG[product_type]['name']}.
        
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
    return base_prompt

def create_test_image():
    """Create a test image with expiry date text for demo purposes"""
    test_img = np.ones((300, 400, 3), dtype=np.uint8) * 255  # White background
    
    # Add some text to simulate expiry dates
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(test_img, "EXP: 01/2026", (50, 50), font, 0.7, (0, 0, 0), 2)
    cv2.putText(test_img, "MFG: 01/2024", (50, 100), font, 0.7, (0, 0, 0), 2)
    cv2.putText(test_img, "Best Before: 31/12/2025", (50, 150), font, 0.7, (0, 0, 0), 2)
    
    return test_img

def capture_image():
    """Capture or upload an image"""
    # Check if we already have the maximum number of images
    if len(st.session_state.captured_images) >= 5:
        st.warning("Maximum 5 images allowed. Please process current images or remove some.")
        return
    
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
    
    # Test image fallback
    st.markdown("---")
    st.write("Or use a test image:")
    
    if st.button("Use Test Image"):
        # Create a test image
        test_img = create_test_image()
        
        # Convert to bytes for session state
        _, buffer = cv2.imencode(".jpg", test_img)
        io_buf = BytesIO(buffer)
        
        # Add to session state
        st.session_state.captured_images.append(test_img)
        st.session_state.captured_image_files.append(io_buf)
        st.rerun()
    
    # Process camera image if available
    if img_file_buffer is not None:
        # Convert to OpenCV format
        bytes_data = img_file_buffer.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Add to session state
        st.session_state.captured_images.append(img)
        st.session_state.captured_image_files.append(img_file_buffer)
        st.rerun()

def display_image_gallery():
    """Display the gallery of captured images"""
    if not st.session_state.captured_images:
        return
    
    st.subheader("Captured Images")
    
    # Create a gallery style container
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
    
    # Create a grid layout
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Button to clear all images
    if st.button("Clear All Images", type="secondary"):
        st.session_state.captured_images = []
        st.session_state.captured_image_files = []
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

def analyze_image(image_file, product_type):
    """Analyze image using Vision API"""
    # For testing/demo purposes without API, generate random results
    current_date = datetime.now().date()
    
    # Create a mock response
    if product_type == "packaged_food":
        exp_date = datetime(2025, 12, 31).date()
        prod_date = datetime(2023, 1, 15).date()
    elif product_type == "dairy":
        exp_date = datetime(2023, 4, 15).date()
        prod_date = datetime(2023, 1, 1).date()
    elif product_type == "medicine":
        exp_date = datetime(2026, 6, 30).date()
        prod_date = datetime(2022, 6, 30).date()
    else:  # cosmetics
        exp_date = datetime(2027, 1, 1).date()
        prod_date = datetime(2022, 1, 1).date()
    
    days_until_expiry = (exp_date - current_date).days
    days_until_prod = (prod_date - current_date).days
    
    mock_result = {
        "dates_found": 2,
        "expiry_dates": [
            {
                "date_text": f"EXP: {exp_date.strftime('%d/%m/%Y')}",
                "date_type": "expiry",
                "standardized_date": exp_date.strftime("%Y-%m-%d"),
                "days_until_expiry": days_until_expiry,
                "expired": days_until_expiry < 0
            },
            {
                "date_text": f"MFG: {prod_date.strftime('%d/%m/%Y')}",
                "date_type": "production",
                "standardized_date": prod_date.strftime("%Y-%m-%d"),
                "days_until_expiry": days_until_prod,
                "expired": days_until_prod < 0
            }
        ],
        "detailed_analysis": f"The image shows a {product_type} package with both expiry and production dates visible."
    }
    
    # In a real application, you would use the Vision API here
    # This is commented out since it requires API keys and external service
    """
    base64_image = encode_image_to_base64(image_file)
    if not base64_image:
        return None
    
    endpoint = get_secret("azure_endpoint")
    api_key = get_secret("azure_api_key")
    model = get_secret("azure_model")
    api_version = get_secret("api_version", "2024-02-15-preview")
    
    if not endpoint or not api_key:
        st.error("Missing API configuration")
        return None
    
    api_url = f"{endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    product_prompt = get_prompt(product_type)
    
    payload = {
        "messages": [
            {"role": "system", "content": f"You are a computer vision assistant specialized in detecting expiry dates."},
            {"role": "user", "content": [
                {"type": "text", "text": product_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        "temperature": 0.5,
        "max_tokens": 800
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            st.error(f"API Error: {response.status_code}")
            return None
        
        result = response.json()
        content = result["choices"][0].get("message", {}).get("content", "{}")
        
        try:
            parsed_result = json.loads(content)
            
            # Process the dates
            current_date = datetime.now().date()
            
            if "expiry_dates" in parsed_result:
                for date_info in parsed_result["expiry_dates"]:
                    if "standardized_date" in date_info:
                        try:
                            expiry_date = datetime.strptime(date_info["standardized_date"], "%Y-%m-%d").date()
                            days_until = (expiry_date - current_date).days
                            date_info["days_until_expiry"] = days_until
                            date_info["expired"] = days_until < 0
                        except Exception:
                            date_info["days_until_expiry"] = 0
                            date_info["expired"] = False
            
            return parsed_result
        except json.JSONDecodeError:
            return None
            
    except requests.exceptions.RequestException:
        return None
    """
    
    # Return mock result for testing
    return mock_result

def process_images():
    """Process all captured images"""
    st.session_state.analysis_results = []
    
    with st.spinner("Analyzing images..."):
        # Process each image
        for idx, img_file in enumerate(st.session_state.captured_image_files):
            # Process the image 
            results = analyze_image(img_file, st.session_state.product_selected)
            
            if results:
                st.session_state.analysis_results.append(results)
    
    # Set flag to show we've processed the images
    st.session_state.images_processed = True

def display_date_card(date_info):
    """Display a single date card with appropriate styling"""
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

def display_results():
    """Display the analysis results"""
    st.header("📊 Analysis Results")
    
    # Process each image result
    for idx, results in enumerate(st.session_state.analysis_results):
        with st.expander(f"Image #{idx+1} Results", expanded=True):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(st.session_state.captured_images[idx], channels="BGR", use_column_width=True)
            
            with col2:
                if results.get("dates_found", 0) > 0 and "expiry_dates" in results:
                    # Display each date card
                    for date_info in results["expiry_dates"]:
                        display_date_card(date_info)
                else:
                    st.warning("No dates detected in this image.")
    
    # Show a summary of all results
    if st.session_state.analysis_results:
        st.header("📝 Summary")
        
        # Get all expiry dates
        all_dates = []
        for result in st.session_state.analysis_results:
            if "expiry_dates" in result:
                all_dates.extend(result["expiry_dates"])
        
        # Filter to just expiry dates (not production)
        expiry_only = [date for date in all_dates if date.get("date_type", "").lower() == "expiry"]
        
        # Count expired, expiring soon, and valid dates
        expired_count = sum(1 for date in expiry_only if date.get("expired", False))
        expiring_soon = sum(1 for date in expiry_only if not date.get("expired", False) and date.get("days_until_expiry", 999) <= 7)
        valid_count = sum(1 for date in expiry_only if not date.get("expired", False) and date.get("days_until_expiry", 0) > 7)
        
        # Create the summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Expired", expired_count)
        with col2:
            st.metric("Expiring Soon", expiring_soon)
        with col3:
            st.metric("Valid", valid_count)
        
        # Display earliest and latest expiry dates
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
                    display_date_card(earliest)
                
                with col2:
                    st.markdown("**Latest Expiry Date:**")
                    display_date_card(latest)
        
        # Start over button
        if st.button("Start Over", type="secondary"):
            st.session_state.product_selected = None
            st.session_state.analysis_results = []
            st.session_state.captured_images = []
            st.session_state.captured_image_files = []
            st.session_state.images_processed = False
            st.rerun()

# Main app flow
def main():
    # If product is not selected, show product selection tiles
    if st.session_state.product_selected is None:
        st.subheader("Select product type")
        
        # Create a grid of product selection tiles
        cols = st.columns(len(PRODUCT_CONFIG))
        
        for i, (product_id, product_data) in enumerate(PRODUCT_CONFIG.items()):
            with cols[i]:
                st.write(f"**{product_data['name']}**", unsafe_allow_html=True)
                
                # Create a clickable button with emoji
                if st.button(f"{product_data['emoji']} Select", key=f"select_{product_id}"):
                    st.session_state.product_selected = product_id
                    st.rerun()
    
    # If product is selected, show the camera interface
    else:
        product_data = PRODUCT_CONFIG[st.session_state.product_selected]
        
        # Header with back button
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
            st.subheader(f"{product_data['emoji']} {product_data['name']} Expiry Detector")
        
        # Only show camera if we're not showing results
        if not st.session_state.images_processed:
            capture_image()
            display_image_gallery()
            
            # Process button
            if st.session_state.captured_images:
                if st.button("Analyze Expiry Dates", type="primary"):
                    process_images()
        
        # Display results if available
        if st.session_state.images_processed and st.session_state.analysis_results:
            display_results()

if __name__ == "__main__":
    main()
