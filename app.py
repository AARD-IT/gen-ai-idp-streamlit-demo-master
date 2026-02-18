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

from PIL import Image
import fitz

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# --- 1. Structured Output Schema ---
class TriageAndExtractInfo(BaseModel):
    """Initial triage and dynamic extraction from a document."""
    document_type: str = Field(description="The type of document, e.g., 'Invoice', 'Receipt', 'Business Card', 'ID Card', 'Other'.")
    signature_present: bool = Field(False, description="True if a signature is found on the document, False otherwise.")
    raw_text: str = Field(description="All raw text content from the document.")
    extracted_data_json: str = Field(description="A JSON formatted string of all relevant key-value pairs extracted from the document.")

# --- 2. LLM Configuration ---
API_KEY = os.environ.get("OPENAI_API_KEY", "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")

llm_global = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=API_KEY,
    max_retries=3,
    timeout=60
)

@st.cache_resource
def get_llm_chain():
    return llm_global.with_structured_output(TriageAndExtractInfo)

# --- 3. Helper Functions — UNCHANGED ---

def encode_image_to_base64(file_path: str) -> Optional[str]:
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
    content_parts = [{"type": "text", "text": instruction}]
    if "text_content" in input_data and input_data["text_content"]:
        content_parts.append({"type": "text", "text": "Raw Text Content: \n" + input_data["text_content"]})
    if "image_url" in input_data:
        content_parts.append({"type": "image_url", "image_url": {"url": input_data["image_url"]}})
    return [HumanMessage(content=content_parts)]

def verify_signature_with_llm(llm, image_data: bytes) -> bool:
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
    SIGNATURE_TEMP_FOLDER = os.path.join(tempfile.gettempdir(), "extracted_signatures")
    try:
        os.makedirs(SIGNATURE_TEMP_FOLDER, exist_ok=True)
        nparr = np.frombuffer(image_data, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h, w, _ = original_img.shape
        roi = original_img[int(h * 0.4):, :]
        roi_y_offset = int(h * 0.4)

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

        if verify_signature_with_llm(llm_global, cropped_bytes):
            signature_file_name = f"signature_{os.path.splitext(file_name)[0]}.png"
            output_path = os.path.join(SIGNATURE_TEMP_FOLDER, signature_file_name)
            cv2.imwrite(output_path, cropped_image)
            return True, output_path
        else:
            return False, None

    except Exception as e:
        print(f"Error during signature extraction/verification: {e}")
        return False, None


def process_document(file_path: str) -> Optional[Dict[str, Any]]:
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

    final_signature_status = result.signature_present
    signature_save_path = None

    if result.signature_present and "image_url" in input_data:
        image_bytes = base64.b64decode(input_data["image_url"].split(",", 1)[1])
        is_verified, save_path = extract_and_save_signature_with_cv(image_bytes, os.path.basename(file_path), extracted_raw_text)
        final_signature_status = is_verified
        signature_save_path = save_path

    return {
        "file_name": os.path.basename(file_path),
        "document_type": result.document_type,
        "signature_present": final_signature_status,
        "extracted_data": extracted_data,
        "signature_path": signature_save_path
    }


# --- Display Function — UNCHANGED ---
def display_results_in_streamlit(data: Dict[str, Any]):
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Document Triage 📄")
        st.info(f"**Document Type:** {data['document_type']}")

        signature_path = data.get('signature_path')

        if data['signature_present'] and signature_path:
            st.success("✅ Signature Found & Verified")
            try:
                isolated_signature = Image.open(signature_path)
                st.image(isolated_signature, caption="Isolated Signature Proof", width=250)
                os.remove(signature_path)
                st.info("Isolated signature image file cleaned up from temp directory.")
            except Exception as e:
                st.error(f"Error displaying signature: Could not access temp file.")
        else:
            st.error("❌ Signature Not Found/Unverified")

    with col2:
        st.subheader("Extracted Dynamic Data 📊")
        st.json(data["extracted_data"], expanded=True)


# =============================================================================
# MAIN APP
# =============================================================================
def main_streamlit_app():
    st.set_page_config(
        page_title="Analytics Avenue - IDP",
        layout="wide",
    )

    # ── GLOBAL CSS ───────────────────────────────────────────
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
        }

        .block-container {
            padding-top: 2rem !important;
            padding-left: 3rem !important;
            padding-right: 3rem !important;
            max-width: 100% !important;
        }

        #MainMenu, footer, header { visibility: hidden; }

        /* Brand */
        .brand-wrap {
            display: flex;
            align-items: center;
            gap: 18px;
            margin-bottom: 28px;
        }
        .brand-name {
            font-size: 26px;
            font-weight: 800;
            color: #064b86;
            line-height: 1.3;
        }
        .divider {
            border: none;
            border-top: 2px solid #e0e0e0;
            margin: 0 0 32px 0;
        }

        /* Page title */
        h1 {
            font-size: 48px !important;
            font-weight: 900 !important;
            color: #0a0a0a !important;
            letter-spacing: -1px !important;
            line-height: 1.1 !important;
            margin-bottom: 6px !important;
        }

        /* Subtitle */
        .subtitle {
            font-size: 17px;
            font-weight: 500;
            color: #555;
            margin-bottom: 36px;
        }

        /* Headings */
        h2 {
            font-size: 30px !important;
            font-weight: 800 !important;
            color: #0a0a0a !important;
            margin-bottom: 16px !important;
        }
        h3 {
            font-size: 22px !important;
            font-weight: 700 !important;
            color: #0a0a0a !important;
            margin-bottom: 12px !important;
        }

        /* Overview cards */
        .card {
            background: #fff;
            border: 1.5px solid #e5e7eb;
            border-radius: 10px;
            padding: 24px 28px;
            margin-bottom: 20px;
        }
        .card-label {
            font-size: 13px;
            font-weight: 700;
            color: #064b86;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .card-text {
            font-size: 16px;
            font-weight: 500;
            color: #222;
            line-height: 1.7;
        }
        .card ul {
            margin: 0;
            padding-left: 18px;
        }
        .card ul li {
            font-size: 15px;
            font-weight: 500;
            color: #333;
            margin-bottom: 6px;
            line-height: 1.6;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0px;
            border-bottom: 2px solid #e0e0e0;
            margin-bottom: 32px;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 16px !important;
            font-weight: 600 !important;
            color: #555 !important;
            padding: 12px 28px !important;
            border: none !important;
            background: transparent !important;
        }
        .stTabs [aria-selected="true"] {
            color: #064b86 !important;
            font-weight: 800 !important;
            border-bottom: 3px solid #064b86 !important;
        }

        /* Form labels */
        .stTextInput label,
        .stSelectbox label,
        .stFileUploader label,
        .stRadio label {
            font-size: 15px !important;
            font-weight: 700 !important;
            color: #0a0a0a !important;
        }

        /* Buttons */
        .stButton > button {
            background-color: #064b86 !important;
            color: #fff !important;
            font-size: 16px !important;
            font-weight: 700 !important;
            padding: 12px 32px !important;
            border-radius: 6px !important;
            border: none !important;
        }
        .stButton > button:hover {
            background-color: #053d70 !important;
        }

        /* Expander */
        .streamlit-expanderHeader p {
            font-size: 16px !important;
            font-weight: 700 !important;
            color: #0a0a0a !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── BRAND HEADER ─────────────────────────────────────────
    logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
    st.markdown(f"""
    <div class="brand-wrap">
        <img src="{logo_url}" width="64" style="border-radius:8px;">
        <div class="brand-name">
            Analytics Avenue &amp;<br>Advanced Analytics
        </div>
    </div>
    <hr class="divider">
    """, unsafe_allow_html=True)

    # ── PAGE TITLE ───────────────────────────────────────────
    st.title("🤖 Gen AI Intelligent Document Processor (IDP)")
    st.markdown('<p class="subtitle">Multi-Modal Document Triage, Extraction, and Signature Verification</p>', unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["Overview", "Application"])

    # ════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ════════════════════════════════════════════════════════
    with tab1:
        st.header("Overview")

        st.markdown("""
        <div class="card">
            <div class="card-label">Purpose</div>
            <div class="card-text">
                Automate the ingestion and understanding of unstructured business documents — invoices, receipts,
                ID cards, contracts, and more — using GPT-4o vision and structured AI extraction pipelines,
                with built-in OpenCV-based signature detection and LLM verification.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.subheader("Capabilities")
            st.markdown("""
            <div class="card">
                <ul>
                    <li>Supports PDF and image files (JPG, PNG, GIF) as input.</li>
                    <li>Automatically classifies document type — Invoice, Receipt, ID Card, Resume, and more.</li>
                    <li>Extracts all key-value pairs dynamically using GPT-4o multimodal vision.</li>
                    <li>Detects and isolates signatures using OpenCV contour analysis.</li>
                    <li>Verifies detected signatures with a secondary LLM confirmation step.</li>
                    <li>Batch processing — upload and analyze multiple documents in one run.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.subheader("Business Impact")
            st.markdown("""
            <div class="card">
                <ul>
                    <li>Eliminate manual data entry from invoices, receipts, and business documents.</li>
                    <li>Accelerate document verification workflows with automated signature detection.</li>
                    <li>Reduce processing time from hours to seconds per document.</li>
                    <li>Standardize extraction output for downstream ERP or database ingestion.</li>
                    <li>Scalable across finance, legal, HR, and procurement document pipelines.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # TAB 2 — APPLICATION (original logic, untouched)
    # ════════════════════════════════════════════════════════
    with tab2:

        mode = st.radio(
            "Select Input Mode:",
            ("Upload Your Documents", "Run Demo (Uses Internal Samples)"),
            horizontal=True
        )

        files_to_process = []

        # Upload Mode
        if mode == "Upload Your Documents":
            uploaded_files = st.file_uploader(
                "Upload single files or select a batch of files (PDF, JPG, PNG, GIF):",
                type=["pdf", "jpg", "jpeg", "png", "gif"],
                accept_multiple_files=True
            )
            if not uploaded_files:
                st.info("⬆️ Please upload documents to start processing.")
            else:
                with tempfile.TemporaryDirectory() as temp_dir:
                    for uploaded_file in uploaded_files:
                        if uploaded_file.size == 0:
                            st.error(f"File {uploaded_file.name} is empty and cannot be processed.")
                            continue
                        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        files_to_process.append((temp_file_path, uploaded_file.name))

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
                                    st.error(f"❌ Failed to process **{file_name}**.")

                        st.balloons()
                        st.success("✅ All documents processed successfully!")
                        st.markdown("---")

        # Demo Mode
        elif mode == "Run Demo (Uses Internal Samples)":
            DEMO_ROOT_FOLDER = "demo_documents"
            NESTED_DEMO_FOLDER = os.path.join(DEMO_ROOT_FOLDER, "ocr_sample_pdf")

            if os.path.isdir(NESTED_DEMO_FOLDER):
                DEMO_FOLDER = NESTED_DEMO_FOLDER
            elif os.path.isdir(DEMO_ROOT_FOLDER):
                DEMO_FOLDER = DEMO_ROOT_FOLDER
            else:
                st.error(f"❌ Error: Demo folders were not found.")
                return

            supported_extensions = ('.pdf', '.jpg', '.jpeg', '.png', '.gif')

            try:
                demo_file_names = os.listdir(DEMO_FOLDER)
                demo_paths = [
                    (os.path.join(DEMO_FOLDER, f), f) for f in demo_file_names
                    if os.path.isfile(os.path.join(DEMO_FOLDER, f)) and f.lower().endswith(supported_extensions)
                ]

                if not demo_paths:
                    st.error(f"No supported demo documents found in '{DEMO_FOLDER}'.")
                    return

                st.success(f"Running Demo with {len(demo_paths)} internal sample documents from '{DEMO_FOLDER}'.")
                st.header("⚡ Processing Results")
                results_container = st.container()

                for file_path, file_name in demo_paths:
                    with results_container.expander(f"**Processing {file_name}**", expanded=False):
                        with st.spinner(f"Analyzing **{file_name}** with Gen AI..."):
                            processed_data = process_document(file_path)
                        if processed_data:
                            display_results_in_streamlit(processed_data)
                        else:
                            st.error(f"❌ Failed to process **{file_name}**.")

                st.balloons()
                st.success("✅ All demo documents processed successfully!")
                st.markdown("---")

            except FileNotFoundError:
                st.error(f"❌ Internal Error: The expected demo folder '{DEMO_FOLDER}' was not accessible.")
                return


if __name__ == "__main__":
    if not API_KEY or "OPENROUTER" in API_KEY:
        st.error("API_KEY is not set or appears to be a placeholder.")
    else:
        main_streamlit_app()
