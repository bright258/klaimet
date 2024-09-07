import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import shap
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from api_server import Base, emissions
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import pipeline
import random
from datetime import datetime

class CO2EmissionAnalyzer:
    def __init__(self, industry, custom_features=None):
        self.industry = industry
        self.custom_features = custom_features or []
        self.model = None
        self.industry_benchmarks = self.load_industry_benchmarks()
        self.regulations = self.load_industry_regulations()
        self.best_practices = self.load_industry_best_practices()
        self.engine = create_engine('sqlite:///co2emissions.db')
        Base.metadata.bind = self.engine
        self.DBSession = sessionmaker(bind=self.engine)

        self.industry_trends = {
            "electronic_manufacturing": [
                "IoT integration for real-time energy monitoring",
                "AI-driven process optimization",
                "Adoption of circular economy principles",
                "Increased use of biodegradable materials",
                "Implementation of digital twins for efficiency"
            ]
        }

    def load_industry_benchmarks(self):
        benchmarks = {
            'electronic_manufacturing': {
                'co2_emissions': 100,
                'energy_efficiency_ratio': 0.8,
            },
            'oil_and_gas': {
                'co2_emissions': 1000,
                'emission_intensity': 0.1,
            }
        }
        return benchmarks.get(self.industry, {})

    def load_industry_regulations(self):
        regulations = {
            'electronic_manufacturing': {
                'max_energy_consumption': 1000,
            },
            'oil_and_gas': {
                'max_co2_emissions': 1500,
            }
        }
        return regulations.get(self.industry, {})

    def load_industry_best_practices(self):
        best_practices = {
            'electronic_manufacturing': {
                'energy_efficiency_ratio': "Implement energy-efficient manufacturing processes and equipment",
            },
            'oil_and_gas': {
                'emission_intensity': "Implement advanced emission capture and storage technologies",
            }
        }
        return best_practices.get(self.industry, {})

    def fetch_data_from_db(self):
        session = self.DBSession()
        emissions_data = session.query(emissions)
        
        data = []
        for emission in emissions_data:
            row = {
                'date': emission.timestamp,
                'co2_emissions': emission.co2_emissions,
                'energy_consumption': emission.energy_consumption,
                'production_volume': emission.production_volume,
            }
            # Add equipment data
            # for equipment in emission.equipment:
            #     row[f'equipment_{equipment.equipment_id}_emissions'] = co2_emissions
            #     row[f'equipment_{equipment.equipment_id}_energy'] = equipment.energy_consumption
            
            # Add process data
            # for process in emission.processes:
            #     row[f'process_{process.process_id}_emissions'] = process.co2_emissions
            #     row[f'process_{process.process_id}_energy'] = process.energy_consumption
            
            # Add energy source data
            # for source in emission.energy_sources:
            #     row[f'source_{source.source_id}_emissions'] = source.co2_emissions
            #     row[f'source_{source.source_id}_energy'] = source.energy_provided
            
            data.append(row)
        
        session.close()
        return pd.DataFrame(data)

    def preprocess_data(self, df):
        df['Date'] = pd.to_datetime(df['date'])
        df.set_index('Date', inplace=True)
        df = df.drop('date', axis=1)  # Remove the original 'date' column
        
        # Convert all columns to numeric, replacing non-numeric values with NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any rows with NaN values
        df = df.dropna()
        
        if self.industry == 'electronic_manufacturing':
            df['energy_efficiency_ratio'] = df['energy_consumption'] / df['production_volume']
        elif self.industry == 'oil_and_gas':
            df['emission_intensity'] = df['co2_emissions'] / df['production_volume']
        
        for feature in self.custom_features:
            if feature not in df.columns:
                print(f"Warning: Custom feature '{feature}' not found in dataset")
        
        return df

    def train_model(self, X, y):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)

    def analyze_feature_importance(self, X):
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values).mean(axis=0)
            feature_importance = dict(zip(X.columns, np.abs(shap_values).mean(0)))
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            major_sources = []
            for feature, importance in sorted_features[:3]:  # Top 3 features
                if importance > 0.1:  # Arbitrary threshold
                    major_sources.append((feature, importance))
            
            if not major_sources:
                print("Warning: No major sources of CO2 emissions identified.")
                major_sources.append(("No significant source", 0))
            
            return major_sources
        except Exception as e:
            print(f"Error in analyzing feature importance: {str(e)}")
            return [("Error in analysis", 0)]

    def analyze_emission_sources(self, X, shap_values):
        try:
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values).mean(axis=0)
            feature_importance = dict(zip(X.columns, np.abs(shap_values).mean(0)))
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            major_sources = []
            for feature, importance in sorted_features[:3]:  # Top 3 features
                if importance > 0.1:  # Arbitrary threshold
                    major_sources.append((feature, importance))
            
            if not major_sources:
                print("Warning: No major sources of CO2 emissions identified.")
                major_sources.append(("No significant source", 0))
            
            return major_sources
        except Exception as e:
            print(f"Error in analyzing emission sources: {str(e)}")
            return [("Error in analysis", 0)]

    def generate_industry_specific_recommendations(self, major_sources):
        recommendations = []
        for source, importance in major_sources:
            if source != "No significant source":
                recommendations.append(f"Reduce emissions from {source}")
        
        if not recommendations:
            recommendations.append("Implement general energy efficiency measures")
        
        return recommendations

    def refine_recommendations_with_custom(self, recommendations):
        industry_specific_recommendations = {
            "electronic_manufacturing": [
                "Implement energy-efficient manufacturing processes",
                "Upgrade to more energy-efficient equipment",
                "Optimize clean room energy consumption",
                "Implement waste heat recovery systems",
                "Use renewable energy sources for manufacturing operations",
                "Improve supply chain sustainability",
                "Implement a comprehensive energy management system",
                "Conduct regular energy audits",
                "Train employees on energy-saving practices",
                "Invest in research and development for more energy-efficient products"
            ]
        }

        refined_recommendations = []
        for rec in recommendations:
            refined_rec = f"Expand on: {rec}\n"
            specific_recs = random.sample(industry_specific_recommendations[self.industry], 3)
            for i, specific_rec in enumerate(specific_recs, 1):
                refined_rec += f"  {i}. {specific_rec}\n"
            refined_recommendations.append(refined_rec)

        return refined_recommendations

    def plot_emissions_trend(self, df):
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['co2_emissions'])
        plt.title('CO2 Emissions Trend')
        plt.xlabel('Date')
        plt.ylabel('CO2 Emissions')
        plt.savefig(f'{self.industry}_emissions_trend.png')
        plt.close()

    def plot_major_sources(self, major_sources):
        if not major_sources:
            print("No major sources to plot.")
            return
        
        sources, importances = zip(*major_sources)
        plt.figure(figsize=(10, 6))
        plt.bar(sources, importances)
        plt.title('Major Sources of CO2 Emissions')
        plt.xlabel('Source')
        plt.ylabel('Importance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{self.industry}_major_sources.png')
        plt.close()

    def plot_benchmark_comparison(self, df):
        metrics = list(self.industry_benchmarks.keys())
        company_values = [df[metric].mean() if metric in df.columns else 0 for metric in metrics]
        industry_values = list(self.industry_benchmarks.values())

        x = range(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x, company_values, width, label='Company')
        ax.bar([i + width for i in x], industry_values, width, label='Industry Average')

        ax.set_ylabel('Value')
        ax.set_title('Company vs Industry Benchmarks')
        ax.set_xticks([i + width/2 for i in x])
        ax.set_xticklabels(metrics)
        ax.legend()

        plt.savefig(f'{self.industry}_benchmark_comparison.png')
        plt.close()

    def generate_report(self, df, major_sources, recommendations):
        report = f"CO2 Emissions Reduction Report for {self.industry.capitalize()} Industry\n\n"
        
        report += "1. Current Emissions Status:\n"
        report += f"   - Average daily CO2 emissions: {df['co2_emissions'].mean():.2f} units\n"
        report += f"   - Emissions trend: {'Increasing' if df['co2_emissions'].iloc[-1] > df['co2_emissions'].iloc[0] else 'Decreasing'}\n"
        report += "   - See 'emissions_trend.png' for a visual representation of the emissions trend.\n"
        
        report += "\n2. Major Sources of CO2 Emissions:\n"
        for source, importance in major_sources:
            report += f"   - {source}: {importance:.2f} importance\n"
        report += "   - See 'major_sources.png' for a visual representation of the major emission sources.\n"
        
        report += "\n3. Industry Benchmarks:\n"
        for metric, value in self.industry_benchmarks.items():
            company_value = df[metric].mean() if metric in df.columns else "N/A"
            report += f"   - {metric}: Company: {company_value:.2f}, Industry Average: {value}\n"
        report += "   - See 'benchmark_comparison.png' for a visual comparison with industry benchmarks.\n"
        
        report += "\n4. Regulatory Compliance:\n"
        for regulation, limit in self.regulations.items():
            if regulation in df.columns:
                compliance = "In Compliance" if df[regulation].max() <= limit else "Non-Compliant"
                report += f"   - {regulation}: {compliance} (Limit: {limit}, Max Value: {df[regulation].max():.2f})\n"
        
        report += "\n5. Recommended Emission Reduction Strategies:\n"
        for i, rec in enumerate(recommendations, 1):
            report += f"   {i}. {rec}\n"
        
        return report