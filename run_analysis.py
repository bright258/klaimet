import pandas as pd
import requests
from datetime import datetime, timedelta
from co2_emission_analyzer import CO2EmissionAnalyzer
import numpy as np
import os
os.environ["OPENAI_API_KEY"] = "***REMOVED***"

def fetch_data_from_api(industry, start_date, end_date):
    API_URL = "http://localhost:8000/emissions/"
    API_KEY = "***REMOVED***"
    
    params = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "industry": industry
    }
    headers = {"X-API-Key": API_KEY}
    
    response = requests.get(API_URL, params=params, headers=headers)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        raise Exception(f"Failed to fetch data from API. Status code: {response.status_code}")

def main(industry, custom_features=None):
    # Remove or comment out this block
    # if not os.getenv("OPENAI_API_KEY"):
    #     print("Warning: OPENAI_API_KEY is not set. GPT-3 refinement will be skipped.")
    
    company_data = {
        'size': 'large',  # or 'small'
        'budget': 'high',  # or 'low'
    }

    analyzer = CO2EmissionAnalyzer(industry, custom_features)
    df = analyzer.fetch_data_from_db()
    df = analyzer.preprocess_data(df)
    
    print("Preprocessed data:")
    print(df.head())
    print(df.dtypes)
    
    if df.empty:
        print("Error: No valid data after preprocessing.")
        return
    
    X = df.drop(['co2_emissions'], axis=1)
    y = df['co2_emissions']
    
    analyzer.train_model(X, y)
    major_sources = analyzer.analyze_feature_importance(X)
    
    if major_sources[0][0] == "Error in analysis":
        print("Error occurred during analysis. Skipping further steps.")
        return
    
    initial_recommendations = analyzer.generate_industry_specific_recommendations(major_sources)
    refined_recommendations = analyzer.refine_recommendations_with_custom(initial_recommendations)
    
    analyzer.plot_emissions_trend(df)
    analyzer.plot_major_sources(major_sources)
    analyzer.plot_benchmark_comparison(df)
    
    report = analyzer.generate_report(df, major_sources, refined_recommendations)
    print(report)

    with open(f'{industry}_emissions_report.txt', 'w') as f:
        f.write(report)
    print(f"Report saved to {industry}_emissions_report.txt")
    print("Visualizations saved as PNG files in the current directory.")

if __name__ == "__main__":
    main('electronic_manufacturing', custom_features=['energy_efficiency_ratio'])
    # Uncomment the line below to run for oil and gas industry
    # main('oil_and_gas', custom_features=['emission_intensity'])