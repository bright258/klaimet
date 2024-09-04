import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import shap
import openai
import matplotlib.pyplot as plt
import seaborn as sns

class CO2EmissionAnalyzer:
    def __init__(self, industry, custom_features=None):
        self.industry = industry
        self.custom_features = custom_features or []
        self.model = None
        self.industry_benchmarks = self.load_industry_benchmarks()
        self.regulations = self.load_industry_regulations()
        self.best_practices = self.load_industry_best_practices()

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

    def preprocess_data(self, df):
        df['Date'] = pd.to_datetime(df['timestamp'])
        df.set_index('Date', inplace=True)
        
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
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        return shap_values

    def analyze_emission_sources(self, X, shap_values):
        feature_importance = dict(zip(X.columns, np.abs(shap_values).mean(0)))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        major_sources = []
        for feature, importance in sorted_features[:3]:  # Top 3 features
            if importance > 0.1:  # Arbitrary threshold
                major_sources.append((feature, importance))
        
        return major_sources

    def generate_industry_specific_recommendations(self, major_sources):
        recommendations = []
        
        if self.industry == 'electronic_manufacturing':
            for source, _ in major_sources:
                if source == 'energy_consumption':
                    recommendations.append("Implement energy-efficient manufacturing processes and equipment. Consider upgrading to the latest energy-efficient machinery and optimizing production schedules to reduce energy consumption during peak hours.")
                elif source == 'production_volume':
                    recommendations.append("Optimize production efficiency to reduce emissions per unit. Implement lean manufacturing principles and invest in automation to increase production efficiency while reducing energy consumption and waste.")
        
        elif self.industry == 'oil_and_gas':
            for source, _ in major_sources:
                if source == 'co2_emissions':
                    recommendations.append("Implement advanced emission capture and storage technologies. Invest in carbon capture and storage (CCS) systems to significantly reduce CO2 emissions from production processes.")
                elif source == 'energy_consumption':
                    recommendations.append("Optimize energy usage in extraction and refining processes. Implement energy management systems and invest in more efficient equipment to reduce overall energy consumption.")
        
        return recommendations

    def refine_recommendations_with_gpt3(self, recommendations):
        openai.api_key = 'your-openai-api-key-here'
        
        prompt = f"Given the following emission reduction recommendations for a {self.industry} company:\n\n"
        for i, rec in enumerate(recommendations, 1):
            prompt += f"{i}. {rec}\n"
        prompt += f"\nPlease refine and expand on these recommendations, providing more specific and actionable advice for the {self.industry} industry, including potential technologies, best practices, and implementation strategies:"
        
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.7,
        )
        
        refined_recommendations = response.choices[0].text.strip().split('\n')
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
        sources, importances = zip(*major_sources)
        plt.figure(figsize=(10, 6))
        plt.bar(sources, importances)
        plt.title('Major Sources of CO2 Emissions')
        plt.xlabel('Source')
        plt.ylabel('Importance')
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