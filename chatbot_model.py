import pandas as pd
import google.generativeai as genai
import re
from langdetect import detect, DetectorFactory
import logging
import io
import numpy as np
import json
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DetectorFactory.seed = 0  # to make language detection consistent

# 🔑 Gemini API key
GEMINI_API_KEY = "AIzaSyC0gdJDMyBRYTTvY5Kxp8FT4KUSqThMLk0"
genai.configure(api_key=GEMINI_API_KEY)

# 📦 Load Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")

# Load your dataset and precompute statistics
try:
    df = pd.read_csv('patient_details2.csv')
    logger.info(f"Dataset loaded with {len(df)} rows")
    
    # Pre-compute dataset statistics
    dataset_stats = {
        'total_rows': len(df),
        'unique_doctors': df['doctor_name'].nunique() if 'doctor_name' in df.columns else 0,
        'doctor_names': df['doctor_name'].unique().tolist() if 'doctor_name' in df.columns else [],
        'unique_invoices': df['invoice_number'].nunique() if 'invoice_number' in df.columns else 0,
        'unique_patients': df['patient_name'].nunique() if 'patient_name' in df.columns else 0,
        'unique_mrn': df['mrn_number'].nunique() if 'mrn_number' in df.columns else 0,
        'total_price': df['price'].sum() if 'price' in df.columns else 0
    }
    
    logger.info(f"Dataset statistics: {dataset_stats}")
        
except Exception as e:
    logger.error(f"Error loading dataset: {str(e)}")
    df = pd.DataFrame()
    dataset_stats = {}

# Function to extract entities from query
def extract_entities(query):
    entities = {
        'invoice_numbers': [],
        'mrn_numbers': [],
        'doctor_names': [],
        'patient_names': []
    }
    
    # Extract invoice numbers (INV followed by digits)
    invoice_pattern = r'INV\d+'
    entities['invoice_numbers'] = re.findall(invoice_pattern, query, re.IGNORECASE)
    
    # Extract MRN numbers (6 or 7 digits)
    mrn_pattern = r'\b\d{6,7}\b'
    entities['mrn_numbers'] = re.findall(mrn_pattern, query)
    
    # Extract doctor names (Dr. followed by name)
    doctor_pattern = r'(?:Dr\.?|Doctor)\s+([A-Za-z\s]+)'
    doctor_matches = re.findall(doctor_pattern, query, re.IGNORECASE)
    entities['doctor_names'] = [name.strip() for name in doctor_matches]
    
    # Extract patient names (capitalized words that might be names)
    if not entities['invoice_numbers'] and not entities['mrn_numbers'] and not entities['doctor_names']:
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        potential_names = re.findall(name_pattern, query)
        # Filter out common words that aren't names
        common_words = ['Table', 'The', 'And', 'But', 'For', 'How', 'What', 'When', 'Where', 'Why', 'Who', 'Which', 'Show', 'Me', 'In', 'Form']
        filtered_names = [name for name in potential_names if name not in common_words]
        if filtered_names:
            entities['patient_names'] = filtered_names
    
    return entities

# Function to get relevant rows based on query
def get_relevant_rows(query, df):
    entities = extract_entities(query)
    logger.info(f"Extracted entities: {entities}")
    
    # Start with an empty dataframe
    relevant_rows = pd.DataFrame()
    
    # If invoice numbers found, filter for those invoices
    if entities['invoice_numbers'] and 'invoice_number' in df.columns:
        invoice_filter = df['invoice_number'].isin(entities['invoice_numbers'])
        invoice_rows = df[invoice_filter]
        relevant_rows = pd.concat([relevant_rows, invoice_rows])
        logger.info(f"Found {len(invoice_rows)} rows for invoices: {entities['invoice_numbers']}")
    
    # If MRN numbers found, filter for those MRNs
    if entities['mrn_numbers'] and 'mrn_number' in df.columns:
        mrn_filter = df['mrn_number'].astype(str).isin(entities['mrn_numbers'])
        mrn_rows = df[mrn_filter]
        relevant_rows = pd.concat([relevant_rows, mrn_rows])
        logger.info(f"Found {len(mrn_rows)} rows for MRNs: {entities['mrn_numbers']}")
    
    # If doctor names found, filter for those doctors
    if entities['doctor_names'] and 'doctor_name' in df.columns:
        doctor_pattern = '|'.join(entities['doctor_names'])
        doctor_filter = df['doctor_name'].str.contains(doctor_pattern, case=False, na=False)
        doctor_rows = df[doctor_filter]
        relevant_rows = pd.concat([relevant_rows, doctor_rows])
        logger.info(f"Found {len(doctor_rows)} rows for doctors: {entities['doctor_names']}")
    
    # If patient names found, filter for those patients
    if entities['patient_names'] and 'patient_name' in df.columns:
        patient_pattern = '|'.join(entities['patient_names'])
        patient_filter = df['patient_name'].str.contains(patient_pattern, case=False, na=False)
        patient_rows = df[patient_filter]
        relevant_rows = pd.concat([relevant_rows, patient_rows])
        logger.info(f"Found {len(patient_rows)} rows for patients: {entities['patient_names']}")
    
    # If we found specific rows, return them
    if len(relevant_rows) > 0:
        relevant_rows = relevant_rows.drop_duplicates()
        logger.info(f"Returning {len(relevant_rows)} specific rows")
        return relevant_rows
    
    # For general queries, return a larger random sample
    sample_size = min(300, len(df))  # Keep within token limits
    random_sample = df.sample(sample_size)
    logger.info(f"Returning random sample of {sample_size} rows")
    return random_sample

# Function to handle general queries using pre-computed statistics
def handle_general_query(query):
    query_lower = query.lower()
    
    # Check for greetings
    if query_lower in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']:
        return "Hello! I'm your dental clinic data assistant. How can I help you today?"
    
    # Check for acknowledgments
    if query_lower in ['ok', 'okay', 'thanks', 'thank you', 'alright']:
        return "You're welcome! Is there anything else you'd like to know about the dental clinic data?"
    
    # Check for doctor count queries
    if any(phrase in query_lower for phrase in ['how many doctors', 'doctor count', 'number of doctors', 'doctors available']):
        if dataset_stats.get('unique_doctors', 0) > 0:
            return f"There are {dataset_stats['unique_doctors']} doctors available: {', '.join(dataset_stats['doctor_names'])}."
    
    # Check for doctor verification queries
    if 'doctor' in query_lower and any(phrase in query_lower for phrase in ['check the', 'verify the', 'is the', 'are the']):
        # Extract any number in the query
        numbers = re.findall(r'\b\d+\b', query)
        if numbers:
            count = int(numbers[0])
            actual_count = dataset_stats.get('unique_doctors', 0)
            if count == actual_count:
                return f"Yes, that's correct! We have {actual_count} doctors: {', '.join(dataset_stats['doctor_names'])}."
            else:
                return f"No, that's not correct. We actually have {actual_count} doctors: {', '.join(dataset_stats['doctor_names'])}."
        else:
            if dataset_stats.get('unique_doctors', 0) > 0:
                return f"We have {dataset_stats['unique_doctors']} doctors: {', '.join(dataset_stats['doctor_names'])}."
    
    # Check for total rows query
    if any(phrase in query_lower for phrase in ['total rows', 'how many rows', 'number of rows']):
        if dataset_stats.get('total_rows', 0) > 0:
            return f"The dataset contains {dataset_stats['total_rows']} rows in total."
    
    # Check for total price queries
    if any(phrase in query_lower for phrase in ['total count price', 'total price', 'sum of prices']):
        if dataset_stats.get('total_price', 0) > 0:
            return f"The total price of all treatments in the dataset is {dataset_stats['total_price']}."
    
    # Check for doctor list query
    if any(phrase in query_lower for phrase in ['list of doctors', 'doctor names', 'show doctors']):
        if dataset_stats.get('doctor_names', []):
            return f"The available doctors are: {', '.join(dataset_stats['doctor_names'])}."
    
    # Check for help queries
    if any(phrase in query_lower for phrase in ['help', 'what can you do', 'how to use']):
        return "I can help you with information about patients, doctors, invoices, and treatments in the dental clinic dataset. You can ask about specific patients, doctors, invoice numbers, or general statistics about the clinic."
    
    # Check for "how you check" type queries
    if any(phrase in query_lower for phrase in ['how you check', 'how do you know', 'how did you find']):
        return "I look through all the patient records in our system and count the unique doctor names to get the accurate count."
    
    # If no general query pattern matches, return None
    return None

# ✅ Urdu/Roman Urdu detection
def is_urdu(text):
    try:
        lang = detect(text)
    except:
        lang = ""
    urdu_chars = re.findall(r'[\u0600-\u06FF]', text)
    has_urdu_script = len(urdu_chars) > 5
    is_probably_roman_urdu = lang in ["ur", "hi", "fa"]
    return has_urdu_script or is_probably_roman_urdu

# ---------- Formatting Functions ----------
def format_response_table(response_text: str):
    """
    Converts markdown or tab-separated table to styled HTML table.
    Removes markdown separator lines with only dashes.
    """
    table_lines = []
    in_table = False
    for line in response_text.splitlines():
        if "|" in line or "\t" in line:
            if re.match(r"^\s*[-\s|]+\s*$", line):  # Skip separator row
                continue
            table_lines.append(line.strip())
            in_table = True
        elif in_table and line.strip() == "":
            break  # End of table
    if table_lines:
        # Detect separator type
        sep = "|" if "|" in table_lines[0] else "\t"
        # Normalize rows
        normalized_lines = []
        for line in table_lines:
            if sep == "|":
                cells = [cell.strip() for cell in line.strip('|').split('|')]
            else:
                cells = [cell.strip() for cell in line.split('\t')]
            normalized_lines.append('\t'.join(cells))  # Normalize to tab
        fixed_table = "\n".join(normalized_lines)
        try:
            df = pd.read_csv(io.StringIO(fixed_table), sep="\t")
            df = df.dropna(how='all')  # Drop empty rows
            df.columns = [col.strip() for col in df.columns]
            # Generate HTML table with improved styling
            html_table = '''
<style>
.table-container {
    max-width: 100%;
    overflow-x: auto;
    margin: 20px 0;
    font-family: Arial, sans-serif;
}
.solid-table {
    border-collapse: collapse;
    width: 100%;
    background-color: #fff;
    box-shadow: 0 2px 3px rgba(0,0,0,0.1);
    border-radius: 8px;
    overflow: hidden;
}
.solid-table th, .solid-table td {
    border: 1px solid #e0e0e0;
    padding: 12px 15px;
    text-align: left;
}
.solid-table th {
    background-color: #f8f9fa;
    color: #333;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 14px;
}
.solid-table tr:nth-child(even) {
    background-color: #f8f9fa;
}
.solid-table tr:hover {
    background-color: #f1f1f1;
}
.solid-table tr.dash-row {
    display: none; /* hide dashed rows if any */
}
</style>
<div class="table-container">
<table class="solid-table">
<thead><tr>
'''
            # Headers
            for col in df.columns:
                html_table += f"<th>{col}</th>"
            html_table += "</tr></thead><tbody>"
            # Rows
            for _, row in df.iterrows():
                row_values = list(row)
                if all(re.match(r"^-+$", str(cell).strip()) for cell in row_values):
                    html_table += '<tr class="dash-row">'
                else:
                    html_table += '<tr>'
                for cell in row_values:
                    html_table += f"<td>{cell}</td>"
                html_table += "</tr>"
            html_table += "</tbody></table></div>"
            return html_table
        except Exception as e:
            logger.error(f"Error parsing table: {str(e)}")
            return "<pre>" + fixed_table + "</pre>"
    return None

def format_response_list(response_text: str) -> str:
    logger.info(f"Formatting response as list: {response_text[:100]}...")
    if not response_text or response_text.strip() == "":
        return "I'm sorry, I couldn't generate a response. Please try again."
    # Remove markdown code blocks
    response_text = re.sub(r'```.*?```', '', response_text, flags=re.DOTALL)
    # REMOVE BOLD MARKDOWN (**text**) globally!
    response_text = re.sub(r'\*\*(.*?)\*\*', r'\1', response_text)
    # Proceed as before
    records = re.split(r'(?=Patient:|MRN:)', response_text)
    formatted_records = []
    for record in records:
        if not record.strip():
            continue
        record = record.strip()
        record = re.sub(r'^-+\s*', '', record)
        # This regex may be unnecessary for your city list, so just add as bullet
        formatted_records.append(f"- {record}")
    result = '\n'.join(formatted_records).strip()
    return result if result else response_text

def format_response_paragraph(response_text: str) -> str:
    logger.info(f"Formatting response as paragraph: {response_text[:100]}...")
    response_text = re.sub(r'```.*?```', '', response_text, flags=re.DOTALL)
    response_text = re.sub(r'\*\*(.*?)\*\*', r'\1', response_text)
    return response_text.replace("\n", " ").strip()



# ---------- Visualization Detection Functions ----------
def detect_visualization_request(user_message):
    """
    Detects if the user is asking for a chart, graph, or visualization.
    Returns a dict with visualization type and parameters, or None if no visualization requested.
    """
    user_message_lower = user_message.lower()
    
    # Keywords that indicate visualization requests
    chart_keywords = ['chart', 'graph', 'plot', 'visualization', 'visualize', 'show me a', 'display', 'draw', 'give me a']
    chart_types = ['bar', 'pie', 'line', 'doughnut', 'histogram']
    
    # Check if it's a visualization request
    is_viz_request = any(keyword in user_message_lower for keyword in chart_keywords)
    
    if not is_viz_request:
        return None
    
    # Determine chart type
    chart_type = 'bar'  # default
    for ctype in chart_types:
        if ctype in user_message_lower:
            chart_type = ctype
            break
    
    # Specific patterns for common requests
    
    # 1. Patients per doctor
    if any(phrase in user_message_lower for phrase in ['patients per doctor', 'number of patients per doctor', 'patient count by doctor', 'patients by doctor']):
        return {
            'type': 'patients_per_doctor',
            'chart_type': chart_type
        }
    
    # 2. Price/revenue per doctor
    if any(phrase in user_message_lower for phrase in ['price per doctor', 'total price per doctor', 'revenue per doctor', 'earnings per doctor', 'income per doctor']):
        return {
            'type': 'price_per_doctor',
            'chart_type': chart_type
        }
    
    # 3. Treatments count by patient
    if any(phrase in user_message_lower for phrase in ['treatments by patient', 'treatment count by patient', 'treatments per patient', 'procedures by patient']):
        return {
            'type': 'treatments_per_patient',
            'chart_type': chart_type
        }
    
    # 4. Invoices per month/time period
    if any(phrase in user_message_lower for phrase in ['invoices per month', 'monthly invoices', 'invoices by month', 'bills per month']):
        return {
            'type': 'invoices_per_month',
            'chart_type': chart_type
        }
    
    # 5. Patient visits per day/time
    if any(phrase in user_message_lower for phrase in ['visits per day', 'daily visits', 'patient visits per day', 'appointments per day']):
        return {
            'type': 'visits_per_day',
            'chart_type': chart_type
        }
    
    # 6. Treatments/procedures count
    if any(phrase in user_message_lower for phrase in ['treatments count', 'procedure count', 'most common treatments', 'popular treatments']):
        return {
            'type': 'treatments_count',
            'chart_type': chart_type
        }
    
    # 7. Patients by city/location
    if any(phrase in user_message_lower for phrase in ['patients by city', 'patients per city', 'location wise patients', 'city wise patients']):
        return {
            'type': 'patients_by_city',
            'chart_type': chart_type
        }
    
    # 8. Price/revenue analysis
    if any(phrase in user_message_lower for phrase in ['price analysis', 'revenue breakdown', 'income analysis', 'earnings breakdown']):
        return {
            'type': 'price_analysis',
            'chart_type': chart_type
        }
    
    # Default: try to determine from context
    if 'doctor' in user_message_lower and ('patient' in user_message_lower or 'count' in user_message_lower):
        return {
            'type': 'patients_per_doctor',
            'chart_type': chart_type
        }
    elif 'doctor' in user_message_lower and ('price' in user_message_lower or 'revenue' in user_message_lower):
        return {
            'type': 'price_per_doctor',
            'chart_type': chart_type
        }
    elif 'month' in user_message_lower and ('invoice' in user_message_lower or 'bill' in user_message_lower):
        return {
            'type': 'invoices_per_month',
            'chart_type': chart_type
        }
    elif 'day' in user_message_lower and ('visit' in user_message_lower or 'patient' in user_message_lower):
        return {
            'type': 'visits_per_day',
            'chart_type': chart_type
        }
    
    # Fallback: general chart
    return {
        'type': 'general_chart',
        'chart_type': chart_type
    }

def generate_chart_data(df, viz_params):
    """
    Generates chart data based on visualization parameters.
    Returns dict with labels, values, and metadata for frontend chart rendering.
    """
    try:
        viz_type = viz_params['type']
        chart_type = viz_params.get('chart_type', 'bar')
        
        # Ensure we have the required columns
        required_columns = ['patient_name', 'doctor_name', 'Invoice date', 'description', 'price', 'city']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return {'error': f'Missing required columns: {missing_columns}'}
        
        if viz_type == 'patients_per_doctor':
            # Count unique patients per doctor
            patient_counts = df.groupby('doctor_name')['patient_name'].nunique().reset_index()
            patient_counts = patient_counts.sort_values('patient_name', ascending=False)
            
            return {
                'labels': patient_counts['doctor_name'].tolist(),
                'values': patient_counts['patient_name'].tolist(),
                'title': 'Number of Patients per Doctor',
                'chart_type': chart_type,
                'total_patients': df['patient_name'].nunique(),
                'x_label': 'Doctor',
                'y_label': 'Number of Patients'
            }
            
        elif viz_type == 'price_per_doctor':
            # Sum total price/revenue per doctor
            price_per_doctor = df.groupby('doctor_name')['price'].sum().reset_index()
            price_per_doctor = price_per_doctor.sort_values('price', ascending=False)
            
            return {
                'labels': price_per_doctor['doctor_name'].tolist(),
                'values': price_per_doctor['price'].tolist(),
                'title': 'Total Revenue per Doctor',
                'chart_type': chart_type,
                'total_revenue': df['price'].sum(),
                'x_label': 'Doctor',
                'y_label': 'Total Revenue'
            }
            
        elif viz_type == 'treatments_per_patient':
            # Count treatments per patient (top 10-15)
            treatment_counts = df.groupby('patient_name').size().reset_index(name='treatment_count')
            treatment_counts = treatment_counts.sort_values('treatment_count', ascending=False).head(15)
            
            return {
                'labels': treatment_counts['patient_name'].tolist(),
                'values': treatment_counts['treatment_count'].tolist(),
                'title': 'Number of Treatments per Patient (Top 15)',
                'chart_type': chart_type,
                'total_treatments': len(df),
                'x_label': 'Patient',
                'y_label': 'Number of Treatments'
            }
            
        elif viz_type == 'invoices_per_month':
            # Convert Invoice date to datetime and group by month
            df['Invoice date'] = pd.to_datetime(df['Invoice date'], errors='coerce')
            df_filtered = df.dropna(subset=['Invoice date'])
            
            if df_filtered.empty:
                return {'error': 'No valid dates found in Invoice date column'}
            
            # Group by year-month
            df_filtered['year_month'] = df_filtered['Invoice date'].dt.to_period('M')
            monthly_counts = df_filtered.groupby('year_month').size().reset_index(name='invoice_count')
            
            return {
                'labels': [str(period) for period in monthly_counts['year_month']],
                'values': monthly_counts['invoice_count'].tolist(),
                'title': 'Number of Invoices per Month',
                'chart_type': chart_type,
                'total_invoices': len(df_filtered),
                'x_label': 'Month',
                'y_label': 'Number of Invoices'
            }
            
        elif viz_type == 'visits_per_day':
            # Convert Invoice date to datetime and group by day
            df['Invoice date'] = pd.to_datetime(df['Invoice date'], errors='coerce')
            df_filtered = df.dropna(subset=['Invoice date'])
            
            if df_filtered.empty:
                return {'error': 'No valid dates found in Invoice date column'}
            
            # Group by date (last 30 days)
            df_filtered['date_only'] = df_filtered['Invoice date'].dt.date
            daily_counts = df_filtered.groupby('date_only').size().reset_index(name='visit_count')
            daily_counts = daily_counts.sort_values('date_only').tail(30)  # Last 30 days
            
            return {
                'labels': [date.strftime('%Y-%m-%d') for date in daily_counts['date_only']],
                'values': daily_counts['visit_count'].tolist(),
                'title': 'Patient Visits per Day (Last 30 Days)',
                'chart_type': chart_type,
                'total_visits': len(df_filtered),
                'x_label': 'Date',
                'y_label': 'Number of Visits'
            }
            
        elif viz_type == 'treatments_count':
            # Count most common treatments/procedures
            treatment_counts = df['description'].value_counts().head(10).reset_index()
            treatment_counts.columns = ['treatment', 'count']
            
            return {
                'labels': treatment_counts['treatment'].tolist(),
                'values': treatment_counts['count'].tolist(),
                'title': 'Most Common Treatments/Procedures',
                'chart_type': chart_type,
                'total_treatments': len(df),
                'x_label': 'Treatment Type',
                'y_label': 'Number of Times Performed'
            }
            
        elif viz_type == 'patients_by_city':
            # Count patients by city
            city_counts = df.groupby('city')['patient_name'].nunique().reset_index()
            city_counts = city_counts.sort_values('patient_name', ascending=False).head(10)
            
            return {
                'labels': city_counts['city'].tolist(),
                'values': city_counts['patient_name'].tolist(),
                'title': 'Number of Patients by City',
                'chart_type': chart_type,
                'total_patients': df['patient_name'].nunique(),
                'x_label': 'City',
                'y_label': 'Number of Patients'
            }
            
        elif viz_type == 'price_analysis':
            # Revenue breakdown by treatment type
            price_by_treatment = df.groupby('description')['price'].sum().reset_index()
            price_by_treatment = price_by_treatment.sort_values('price', ascending=False).head(10)
            
            return {
                'labels': price_by_treatment['description'].tolist(),
                'values': price_by_treatment['price'].tolist(),
                'title': 'Revenue by Treatment Type',
                'chart_type': chart_type,
                'total_revenue': df['price'].sum(),
                'x_label': 'Treatment Type',
                'y_label': 'Total Revenue'
            }
            
        else:  # general_chart fallback
            # Get top treatments as default
            treatment_counts = df['description'].value_counts().head(8).reset_index()
            treatment_counts.columns = ['treatment', 'count']
            
            return {
                'labels': treatment_counts['treatment'].tolist(),
                'values': treatment_counts['count'].tolist(),
                'title': 'Most Common Treatments',
                'chart_type': chart_type,
                'total_records': len(df),
                'x_label': 'Treatment',
                'y_label': 'Frequency'
            }
            
    except Exception as e:
        logger.error(f"Error generating chart data: {str(e)}")
        return {'error': f'Error generating chart: {str(e)}'}


# ---------- Main Chat Function ----------
def get_chat_response(user_message, df, session_history=None, answer_format='auto'):
    """
    Handles user queries:
    - Directly returns first/last patient names from dataset
    - Uses Gemini AI for all other queries
    - Supports HTML table, list, and paragraph formatting
    - Visualization requests (charts/graphs)
    """
    try:
        msg_lower = user_message.lower()

        # ---------- First/Last Patient Rules ----------
        if "first 3 patients" in msg_lower:
            if "patient_name" in df.columns:
                patients = df["patient_name"].dropna().unique()[:3]
                return "- Here are the first 3 patients:\n" + "\n".join([f"* {p}" for p in patients])
            return "No patient records found."

        if "first patient" in msg_lower:
            if "patient_name" in df.columns:
                patient = df["patient_name"].dropna().unique()[0]
                return f"- The first patient's name is {patient}."
            return "No patient records found."

        if "last patient" in msg_lower:
            if "patient_name" in df.columns:
                patient = df["patient_name"].dropna().unique()[-1]
                return f"- The last patient's name is {patient}."
            return "No patient records found."

        # Check if user is requesting a visualization
        viz_params = detect_visualization_request(user_message)
        if viz_params:
            logger.info(f"Detected visualization request: {viz_params}")
            chart_data = generate_chart_data(df, viz_params)
            
            if 'error' in chart_data:
                return f"I couldn't generate the chart: {chart_data['error']}"
            
            # Return chart data as JSON embedded in a special format
            # The frontend will detect this and render the chart
            chart_json = json.dumps(chart_data)
            return f"CHART_DATA:{chart_json}"
        

        # ---------- General Queries (Handled by pre-computed stats if possible) ----------
        general_response = handle_general_query(user_message)
        if general_response:
            return general_response

        # ---------- Otherwise, get relevant rows and send to Gemini ----------
        df_sample = get_relevant_rows(user_message, df)
        df_sample = df_sample.dropna(how='all', axis=0).dropna(how='all', axis=1)

        columns = df_sample.columns.tolist()
        data_preview = df_sample.to_dict(orient='records')

        urdu_requested = is_urdu(user_message)
        language_instruction = "جواب صرف اردو میں دیں۔ انگریزی استعمال نہ کریں۔\n\n" if urdu_requested else ""

        # Session history (last 5 messages)
        history_text = ""
        if session_history:
            history_text = "\n\nRECENT CHAT HISTORY:\n"
            for user_msg, bot_resp in session_history[-5:]:
                user_msg = user_msg[:200] + "..." if len(user_msg) > 200 else user_msg
                bot_resp = bot_resp[:200] + "..." if len(bot_resp) > 200 else bot_resp
                history_text += f"User: {user_msg}\nBot: {bot_resp}\n\n"

        # Dataset statistics for Gemini prompt
        stats_text = f"""
Here are some key statistics about the dental clinic:
- Total patients: {dataset_stats['unique_patients']}
- Total doctors: {dataset_stats['unique_doctors']}
- Doctor names: {', '.join(dataset_stats['doctor_names'])}
- Total invoices: {dataset_stats['unique_invoices']}
- Total price: {dataset_stats['total_price']}
"""

        # Prompt for Gemini
        prompt = f"""
You are a friendly dental clinic assistant. You help with patient records, appointments, invoices, and treatments. Keep your answers short, friendly, and conversational.

{stats_text}

Columns in dataset: {columns}

Relevant records:
{data_preview}

{history_text}

Please answer this question: "{user_message}"

{language_instruction}

Remember to:
- Keep it short and friendly
- Use tables when showing data
- Be accurate with numbers (use the statistics provided above for totals)
- Don't say "based on the dataset" or mention the dataset
- If you don't know something, just say so politely
"""

        logger.info(f"Sending prompt to Gemini: {prompt[:200]}...")
        response = model.generate_content(prompt)
        logger.info(f"Received response: {response.text[:100]}...")

        # ---------- Formatting ----------
        if answer_format == 'auto' or answer_format == 'table':
            table_html = format_response_table(response.text)
            if table_html:
                return table_html
            if answer_format == 'table':
                return response.text.strip()
            return format_response_list(response.text)
        elif answer_format == 'list':
            return format_response_list(response.text)
        elif answer_format == 'paragraph':
            return format_response_paragraph(response.text)
        else:
            return response.text.strip()

    except Exception as e:
        logger.error(f"Error in get_chat_response: {str(e)}")
        return f"Error generating response: {str(e)}"
