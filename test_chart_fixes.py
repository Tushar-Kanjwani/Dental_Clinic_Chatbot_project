#!/usr/bin/env python3

"""
Test script to verify the chart detection and generation fixes work correctly
with the provided test queries.
"""

import pandas as pd
import sys
import os

# Add the current directory to the path to import chatbot_model
sys.path.append(os.getcwd())

from chatbot_model import detect_visualization_request, generate_chart_data

# Test queries from the user
test_queries = [
    "Show me a bar chart of the number of patients per doctor.",
    "Give me a bar graph of total price per doctor.",
    "Display a bar chart of treatments count by patient.",
    "Show me a bar chart of invoices per month.",
    "Can you make a bar chart of patient visits per day?"
]

def test_chart_detection():
    """Test if the visualization detection works for all test queries"""
    print("🔍 Testing Chart Detection...")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        
        viz_params = detect_visualization_request(query)
        
        if viz_params:
            print(f"   ✅ Detected: {viz_params['type']} ({viz_params['chart_type']} chart)")
        else:
            print(f"   ❌ Not detected as visualization request")
    
    print("\n" + "=" * 50)

def test_chart_generation():
    """Test chart data generation with sample data"""
    print("\n📊 Testing Chart Data Generation...")
    print("=" * 50)
    
    # Create sample data matching the structure of patient_details2.csv
    sample_data = {
        'mrn_number': ['25081021', '25081023', '25081024', '25081022', '25081025'],
        'patient_name': ['HABIB ULLAH KHAN', 'BADSHAH REHMAN', 'AHMED ZIA', 'JIBRAEL KHAN', 'SARA AHMED'],
        'Registration date': ['2025-08-05 14:06:34'] * 5,
        'city': ['ATTOCK', 'islamabad', 'islamabad', 'islamabad', 'lahore'],
        'invoice_number': ['INV25081069', 'INV25081068', 'INV25081067', 'INV25081066', 'INV25081065'],
        'Invoice date': ['2025-08-05 17:18:47', '2025-08-05 17:01:20', '2025-08-05 16:18:59', '2025-08-05 15:52:31', '2025-08-05 14:30:00'],
        'description': ['Composite filling large', 'Root canal treatment Posterior tooth', 'Consultation Fee', 'Root canal treatment Posterior tooth', 'Scaling and polishing per arch'],
        'price': [10000, 33600, 3000, 33600, 12500],
        'doctor_name': ['Dr Saqib', 'Dr Saqib', 'Dr Israr', 'Dr Saqib', 'Dr Israr']
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Sample dataset created with {len(df)} rows")
    
    # Test each query type
    test_viz_params = [
        {'type': 'patients_per_doctor', 'chart_type': 'bar'},
        {'type': 'price_per_doctor', 'chart_type': 'bar'},
        {'type': 'treatments_per_patient', 'chart_type': 'bar'},
        {'type': 'invoices_per_month', 'chart_type': 'bar'},
        {'type': 'visits_per_day', 'chart_type': 'bar'}
    ]
    
    for i, viz_params in enumerate(test_viz_params, 1):
        print(f"\n{i}. Testing {viz_params['type']}:")
        
        try:
            chart_data = generate_chart_data(df, viz_params)
            
            if 'error' in chart_data:
                print(f"   ❌ Error: {chart_data['error']}")
            else:
                print(f"   ✅ Success: {chart_data['title']}")
                print(f"      Labels: {chart_data['labels'][:3]}...")  # Show first 3 labels
                print(f"      Values: {chart_data['values'][:3]}...")  # Show first 3 values
                
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
    
    print("\n" + "=" * 50)

def main():
    print("🚀 Testing Chatbot Chart Fixes")
    print("=" * 50)
    
    # Test detection
    test_chart_detection()
    
    # Test generation
    test_chart_generation()
    
    print("\n✨ Testing Complete!")
    
    # Summary
    print("\n📋 Summary:")
    print("- All test queries should now be properly detected as chart requests")
    print("- Chart data generation should work for all common visualization types")
    print("- The frontend will automatically render these as interactive charts")

if __name__ == "__main__":
    main()
