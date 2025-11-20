import os
import base64
import json
import pypdf
import cv2
import numpy as np
import re
import tempfile
import streamlit as st
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path 
import tempfile
import time


# Required for image cropping and PDF rendering
from PIL import Image
import fitz

# Core LangChain components
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# --- 1. Define the Structured Output Schema ---
class TriageAndExtractInfo(BaseModel):
    """Initial triage and dynamic extraction from a document."""
    document_type: str = Field(description="The type of document, e.g., 'Invoice', 'Receipt', 'Business Card', 'ID Card', 'Other'.")
    signature_present: bool = Field(False, description="True if a signature is found on the document, False otherwise.")
    raw_text: str = Field(description="All raw text content from the document.") 
    extracted_data_json: str = Field(description="A JSON formatted string of all relevant key-value pairs extracted from the document.")

# --- 2. Configure the LLM for your custom endpoint ---
API_KEY = os.environ.get("OPENAI_API_KEY", "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")

# --- Global LLM initialization (used by verify_signature_with_llm directly) ---
llm_global = ChatOpenAI(
    model="gpt-4o-mini",   # lighter, faster model
    temperature=0,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=API_KEY,
    max_retries=3,
    timeout=60
)
# --- 3. Create the Agentic Pipeline with .with_structured_output() ---
@st.cache_resource
def get_llm_chain():
    return llm_global.with_structured_output(TriageAndExtractInfo)

# --- 4. Helper Functions for File Handling and Processing ---

def encode_image_to_base64(file_path: str) -> Optional[str]:
    """Encodes a file (image or PDF) to a base64 string."""
    try:
        if file_path.lower().endswith('.pdf'):
            DPI = 300
            doc = fitz.open(file_path)
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=DPI)
            img_bytes = pix.tobytes("png")
            doc.close()
            return base64.b64encode(img_bytes).decode("utf-8")
        else:
            with open(file_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"An error occurred while encoding the file to base64: {e}")
        return None

def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """Extracts raw text from a text-based PDF."""
    try:
        reader = pypdf.PdfReader(pdf_path)
        text_content = ""
        for page in reader.pages:
            text_content += page.extract_text() or ""
        return text_content
    except Exception as e:
        print(f"An error occurred during raw text extraction: {e}")
        return None

def format_input_to_message(input_data: Dict[str, str], instruction: str) -> List:
    """Formats input into a multimodal HumanMessage with a specific instruction."""
    content_parts = [
        {"type": "text", "text": instruction},
    ]
    if "text_content" in input_data and input_data["text_content"]:
        content_parts.append({"type": "text", "text": "Raw Text Content: \n" + input_data["text_content"]})
    if "image_url" in input_data:
        content_parts.append({"type": "image_url", "image_url": {"url": input_data["image_url"]}})
    return [HumanMessage(content=content_parts)]

def verify_signature_with_llm(llm, image_data: bytes) -> bool:
    """Verifies if the provided image data contains a signature using an LLM."""
    try:
        base64_image = base64.b64encode(image_data).decode("utf-8")
        instruction = "Is the object in this image a signature? Answer only 'Yes' or 'No'."
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ]
        )
        response = llm.invoke([message]).content
        return "yes" in response.strip().lower()

    except Exception as e:
        return False
    
    

def extract_and_save_signature_with_cv(image_data: bytes, file_name: str, raw_text: str):
    """
    Finds and verifies the signature area using OpenCV and LLM. 
    Saves the cropped signature to a temporary folder if verified.
    Returns: Tuple (bool: is_verified, str: save_path or None)
    """

    SIGNATURE_TEMP_FOLDER = os.path.join(tempfile.gettempdir(), "extracted_signatures")

    
    try:
        os.makedirs(SIGNATURE_TEMP_FOLDER, exist_ok=True)
        nparr = np.frombuffer(image_data, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h, w, _ = original_img.shape
        roi = original_img[int(h * 0.4):, :]
        roi_y_offset = int(h * 0.4)

        # --- OpenCV Logic (Contour Finding) ---
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        thresh_roi = cv2.adaptiveThreshold(blur_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        thresh_roi = cv2.morphologyEx(thresh_roi, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(thresh_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate_bbox = None
        best_score = -1

        for c in contours:
            area = cv2.contourArea(c)
            if area < 50: continue
            
            x, y, w_c, h_c = cv2.boundingRect(c)
            aspect_ratio = w_c / h_c if h_c > 0 else 0
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0

            is_valid_area = 200 < area < 50000
            is_signature_like_shape = (1.5 < aspect_ratio < 6.0) and solidity > 0.4
            is_in_lower_half = (y + roi_y_offset) > h * 0.5
            
            text_in_vicinity = False
            signature_keywords = ["signature", "signed", "signatory", "on behalf of", "authorized"]
            for keyword in signature_keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', raw_text.lower()):
                    text_in_vicinity = True
                    break

            score = 0
            if is_valid_area: score += 10
            if is_signature_like_shape: score += 20
            if is_in_lower_half: score += 15
            if text_in_vicinity: score += 50
            score += np.log(area)
            if score > best_score:
                best_score = score
                best_candidate_bbox = (x, y, w_c, h_c)

        if not best_candidate_bbox:
            return False, None

        x, y, w_c, h_c = best_candidate_bbox
        x_final = x
        y_final = y + roi_y_offset
        buffer = 15 
        x_min = max(0, x_final - buffer)
        y_min = max(0, y_final - buffer)
        x_max = min(w, x_final + w_c + buffer)
        y_max = min(h, y_final + h_c + buffer)
        
        cropped_image = original_img[y_min:y_max, x_min:x_max]
        
        is_success, buffer = cv2.imencode(".png", cropped_image)
        if not is_success:
            return False, None
            
        cropped_bytes = buffer.tobytes()
        
        # Final LLM verification
        if verify_signature_with_llm(llm_global, cropped_bytes):
            signature_file_name = f"signature_{os.path.splitext(file_name)[0]}.png"
            output_path = os.path.join(SIGNATURE_TEMP_FOLDER, signature_file_name)
            cv2.imwrite(output_path, cropped_image)
            print(f"SIGNATURE SAVED: {output_path}") 
            return True, output_path
        else:
            return False, None 

    except Exception as e:
        print(f"Error during signature extraction/verification: {e}")
        return False, None


def process_document(file_path: str) -> Optional[Dict[str, Any]]:
    """Main function to process a document (PDF or image)."""
    input_data = {}
    
    is_pdf = file_path.lower().endswith('.pdf')
    is_image = file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))

    if not (is_pdf or is_image):
        return None

    triage_and_extract_llm = get_llm_chain()
    
    base64_image = encode_image_to_base64(file_path)
    if not base64_image:
        return None
    
    input_data["image_url"] = f"data:image/jpeg;base64,{base64_image}"
    raw_text = ""
    if is_pdf:
        raw_text = extract_text_from_pdf(file_path) or ""
        if raw_text:
             input_data["text_content"] = raw_text

    # --- LLM Processing Loop ---
    max_retries = 3
    result = None
    extracted_data = {}

    for attempt in range(max_retries):
        try:
            instruction = """You are a professional document analysis assistant. From the document provided (image and optionally raw text), extract the document type and if a signature is present. Additionally, extract ALL other relevant key-value pairs into the 'extracted_data_json' field.
            - Dates must be in the format 'YYYY-MM-DD'.
            - Monetary amounts must include the currency symbol (e.g., 'USD 100.00').
            - Line items must be a list of dictionaries if present.
            - The 'document_type' should be concise (e.g., 'Invoice', 'ID Card', 'Receipt', or 'Resume')."""
            
            messages = format_input_to_message(input_data, instruction)
            result = triage_and_extract_llm.invoke(messages)
            
            extracted_raw_text = result.raw_text
            extracted_data = json.loads(result.extracted_data_json)
            break 
            
        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                extracted_data = {}
        except Exception:
            if attempt == max_retries - 1:
                return None

    if result is None:
        return None
    
    # Signature Extraction and Verification
    final_signature_status = result.signature_present
    signature_save_path = None
    
    if result.signature_present and "image_url" in input_data:
        image_bytes = base64.b64decode(input_data["image_url"].split(",", 1)[1])
        
        is_verified, save_path = extract_and_save_signature_with_cv(image_bytes, os.path.basename(file_path), extracted_raw_text)
        
        final_signature_status = is_verified
        signature_save_path = save_path

    final_output = {
        "file_name": os.path.basename(file_path),
        "document_type": result.document_type,
        "signature_present": final_signature_status,
        "extracted_data": extracted_data,
        "signature_path": signature_save_path 
    }

    return final_output

# --- 5. Streamlit Application Block ---

def display_results_in_streamlit(data: Dict[str, Any]):
    """Displays the extracted document information in a clean, structured format."""
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Document Triage 📄")
        st.info(f"**Document Type:** {data['document_type']}")
        
        signature_path = data.get('signature_path')
        
        if data['signature_present'] and signature_path:
            st.success("✅ Signature Found & Verified")
            
            # --- CODE ADDED TO DISPLAY THE IMAGE ---
            try:
                # 1. Open the image from the temporary path
                isolated_signature = Image.open(signature_path)
                
                # 2. Display the image
                st.image(isolated_signature, caption="Isolated Signature Proof", width=250)
                
                # 3. Clean up the temporary file (important!)
                os.remove(signature_path)
                st.info("Isolated signature image file cleaned up from temp directory.")
                
            except Exception as e:
                st.error(f"Error displaying signature: Could not access temp file.")
            # --- END ADDED CODE ---
            
        else:
            st.error("❌ Signature Not Found/Unverified")
        
    with col2:
        st.subheader("Extracted Dynamic Data 📊")
        st.json(data["extracted_data"], expanded=True)

def main_streamlit_app():
    st.set_page_config(
        page_title="Analytics Avenue - IDP", 
        layout="wide",
    )
    
    # =========================================================================
    # UPDATED HEADER: Company Logo + Name (Analytics Avenue)
    # =========================================================================
    logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
    st.markdown(f"""
    <div style="display:flex; align-items:center; margin-bottom:20px;">
        <img src="{logo_url}" width="60" style="margin-right:10px;">
        <div style="line-height:1;">
            <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0;">Analytics Avenue &</div>
            <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0;">Advanced Analytics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # App Specific Title (Kept below branding)
    st.title("🤖 Gen AI Intelligent Document Processor (IDP)")
    st.subheader("Multi-Modal Document Triage, Extraction, and Verification")
    
    st.markdown("---")
    
    # 1. Mode Selection
    mode = st.radio(
        "Select Input Mode:",
        ("Upload Your Documents", "Run Demo (Uses Internal Samples)"),
        horizontal=True
    )
    
    files_to_process = []
    
    # 2. Upload Mode
    if mode == "Upload Your Documents":
        # Text updated for clarity on single/batch upload
        uploaded_files = st.file_uploader(
            "Upload single files or select a batch of files (PDF, JPG, PNG, GIF):",
            type=["pdf", "jpg", "jpeg", "png", "gif"],
            accept_multiple_files=True
        )
        if not uploaded_files:
            st.info("⬆️ Please upload documents to start processing.")
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                if uploaded_file.size == 0:
                    st.error(f"File {uploaded_file.name} is empty and cannot be processed.")
                    continue
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                files_to_process.append((temp_file_path, uploaded_file.name))
            
            # 4. Processing Execution (Executed inside the temporary context)
            if files_to_process:
                st.header("⚡ Processing Results")
                results_container = st.container()
                
                for file_path, file_name in files_to_process:
                    with results_container.expander(f"**Processing {file_name}**", expanded=False):
                        with st.spinner(f"Analyzing **{file_name}** with Gen AI..."):
                            processed_data = process_document(file_path) 
                            
                        if processed_data:
                            display_results_in_streamlit(processed_data)
                        else:
                            st.error(f"❌ Failed to process **{file_name}**. Check console logs for errors.")
                
                st.balloons()
                st.success("✅ All documents processed successfully (or finished)! Scroll up to see the results.")
                st.markdown("---")
    
    # 3. Demo Mode Execution (Runs on the packaged files)
    elif mode == "Run Demo (Uses Internal Samples)":
        # Path resolution logic for nested or root demo folders
        DEMO_ROOT_FOLDER = "demo_documents"
        NESTED_DEMO_FOLDER = os.path.join(DEMO_ROOT_FOLDER, "ocr_sample_pdf")
        
        if os.path.isdir(NESTED_DEMO_FOLDER):
            DEMO_FOLDER = NESTED_DEMO_FOLDER
        elif os.path.isdir(DEMO_ROOT_FOLDER):
            DEMO_FOLDER = DEMO_ROOT_FOLDER
        else:
            st.error(f"❌ Error: Demo folders were not found. Ensure 'demo_documents' or 'demo_documents/ocr_sample_pdf' exist.")
            return
            
        supported_extensions = ('.pdf', '.jpg', '.jpeg', '.png', '.gif')
        
        try:
            demo_file_names = os.listdir(DEMO_FOLDER)
            demo_paths = [
                (os.path.join(DEMO_FOLDER, f), f) for f in demo_file_names 
                if os.path.isfile(os.path.join(DEMO_FOLDER, f)) and f.lower().endswith(supported_extensions)
            ]
            
            if not demo_paths:
                st.error(f"No supported demo documents found in the selected folder: '{DEMO_FOLDER}'.")
                return
            
            st.success(f"Running Demo with {len(demo_paths)} internal sample documents from '{DEMO_FOLDER}'.")
            
            # 4. Processing Execution (Directly on demo paths)
            st.header("⚡ Processing Results")
            results_container = st.container()
            
            for file_path, file_name in demo_paths:
                with results_container.expander(f"**Processing {file_name}**", expanded=False):
                    with st.spinner(f"Analyzing **{file_name}** with Gen AI..."):
                        processed_data = process_document(file_path) 
                        
                    if processed_data:
                        display_results_in_streamlit(processed_data)
                    else:
                        st.error(f"❌ Failed to process **{file_name}**. Check console logs for errors.")
            
            st.balloons()
            st.success("✅ All demo documents processed successfully! Scroll up to see the results.")
            st.markdown("---")
        
        except FileNotFoundError:
            st.error(f"❌ Internal Error: The expected demo folder '{DEMO_FOLDER}' was not accessible.")
            return

if __name__ == "__main__":
    if not API_KEY or "OPENROUTER" in API_KEY: 
        st.error("API_KEY is not set or appears to be a placeholder. Please ensure the OPENAI_API_KEY environment variable is correctly configured.")
    else:
        # NOTE: Run this command in your terminal: streamlit run app.py
        main_streamlit_app()
