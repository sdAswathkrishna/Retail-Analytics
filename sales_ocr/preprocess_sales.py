import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import io

import os


def clean_sales_data(sales_data_file):
    """
    Clean and preprocess sales data according to specifications:
    - Remove duplicates (keep first valid entry)
    - Filter sales values to range (0-30000)
    - Fill missing dates with mean values
    """
    
    # Read the CSV data
    df = pd.read_csv(sales_data_file)
    
    print("Original data shape:", df.shape)
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    
    # Filter sales values to be in range (1000-30000)
    df['Index'] = df.index

    df_filtered = df[(df['Amount'] > 1000) & (df['Amount'] <= 30000)].copy()

    # Sort by date to ensure we keep the first occurrence
    df_filtered = df_filtered.sort_values(['Date', 'Index'])  # Secondary sort


    df_filtered = df_filtered.drop(columns=['Index'])

    df_clean = df_filtered.drop_duplicates(subset=['Date'], keep='first')

    # Handle multiple months and years
    # Group by year and month, then fill missing dates for each group
    df_clean['Year'] = df_clean['Date'].dt.year
    df_clean['Month'] = df_clean['Date'].dt.month

    all_results = []
    for (year, month), group in df_clean.groupby(['Year', 'Month']):
        # Get the full date range for this month
        start_date = group['Date'].min().replace(day=1)
        # Find last day of the month
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        end_date = next_month - timedelta(days=1)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        complete_df = pd.DataFrame({'Date': date_range})
        # Merge with group data
        merged = complete_df.merge(group[['Date', 'Amount']], on='Date', how='left')
        # Calculate mean for imputation (only from valid data in this group)
        mean_sales = round(group['Amount'].mean(), -1)
        print(f"\n[{year}-{month:02d}] Mean sales value for imputation: {mean_sales}")
        merged['Amount'] = merged['Amount'].fillna(mean_sales)
        merged['Amount'] = merged['Amount'].astype(int)
        all_results.append(merged)

    # Concatenate all months
    df_final = pd.concat(all_results, ignore_index=True)
    # Rename columns for clarity
    df_final = df_final.rename(columns={'Amount': 'Sales'})
    # Format date for better readability
    df_final['Date'] = df_final['Date'].dt.strftime('%Y-%m-%d')
    return df_final

def main():
    """Main function to execute the data cleaning process"""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "ocr_sales_data.csv")
    # Clean the data
    cleaned_data = clean_sales_data(csv_path)
    
    # Save to CSV
    output_filename = 'cleaned_sales_data.csv'
    cleaned_data.to_csv(output_filename, index=False)
    print(f"\nCleaned data saved to: {output_filename}")
    
    # Additional validation
    print("\n" + "="*50)
    print("DATA VALIDATION")
    print("="*50)
    
    # Check for duplicates
    duplicates = cleaned_data['Date'].duplicated().sum()
    print(f"Duplicate dates: {duplicates}")
    
    # Check sales range
    out_of_range = cleaned_data[(cleaned_data['Sales'] <= 1000) | (cleaned_data['Sales'] > 30000)].shape[0]
    print(f"Sales values outside range (1000-30000): {out_of_range}")
    
    # Check for missing values
    missing_values = cleaned_data.isnull().sum().sum()
    print(f"Missing values: {missing_values}")
    
    print("\nData cleaning completed successfully!")
    
    return cleaned_data

if __name__ == "__main__":
    # Execute the main function
    cleaned_df = main()