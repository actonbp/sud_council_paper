#!/usr/bin/env python3
"""
05_demographic_interest_analysis.py
Purpose: Merge demographic survey data with LLM interest labels and explore relationships
Author: AI Assistant, 2025-08-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

def load_and_merge_data():
    """Load demographic data and merge with LLM interest labels"""
    
    # Load LLM labels
    llm_df = pd.read_csv('results/study2/clean_participant_dataset_fewshot.csv')
    print(f"Loaded {len(llm_df)} participants with LLM labels")
    
    # Load demographic data
    demo_df = pd.read_csv('data/focusgroup-survey-raw.csv')
    print(f"Loaded {len(demo_df)} participants with demographic data")
    
    # Rename ID column for consistency
    demo_df = demo_df.rename(columns={'ID#': 'participant_id'})
    
    # Merge on participant_id
    merged_df = llm_df[['participant_id', 'ai_label']].merge(
        demo_df, 
        on='participant_id', 
        how='inner'
    )
    
    print(f"\nMerged {len(merged_df)} participants with both LLM labels and demographics")
    print(f"Lost {len(llm_df) - len(merged_df)} participants without demographic data")
    
    return merged_df

def explore_categorical_relationships(df, var_name, var_label):
    """Explore relationship between a categorical variable and interest"""
    
    # Create crosstab
    ct = pd.crosstab(df[var_name], df['ai_label'])
    
    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(ct)
    
    # Calculate percentages
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    
    print(f"\n{var_label}:")
    print("Crosstab (counts):")
    print(ct)
    print("\nPercentages (row %):")
    print(ct_pct.round(1))
    print(f"\nChi-square test: χ² = {chi2:.3f}, p = {p_value:.3f}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Stacked bar chart
    ct_pct.plot(kind='bar', stacked=True, ax=ax1)
    ax1.set_title(f'{var_label} by Interest Level')
    ax1.set_xlabel(var_label)
    ax1.set_ylabel('Percentage')
    ax1.legend(title='Interest', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Grouped bar chart
    ct.plot(kind='bar', ax=ax2)
    ax2.set_title(f'{var_label} by Interest Level (Counts)')
    ax2.set_xlabel(var_label)
    ax2.set_ylabel('Count')
    ax2.legend(title='Interest', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'results/study2/demographic_{var_name}_interest.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return chi2, p_value

def explore_continuous_relationships(df, var_name, var_label):
    """Explore relationship between a continuous variable and interest"""
    
    # Convert to numeric, handling errors
    df[var_name] = pd.to_numeric(df[var_name], errors='coerce')
    
    # Remove missing values
    df_clean = df.dropna(subset=[var_name])
    
    # Group statistics
    stats_summary = df_clean.groupby('ai_label')[var_name].describe()
    print(f"\n{var_label} by Interest:")
    print(stats_summary)
    
    # T-test
    interested = df_clean[df_clean['ai_label'] == 'INTERESTED'][var_name]
    not_interested = df_clean[df_clean['ai_label'] == 'NOT_INTERESTED'][var_name]
    
    t_stat, p_value = stats.ttest_ind(interested, not_interested)
    print(f"\nT-test: t = {t_stat:.3f}, p = {p_value:.3f}")
    
    # Cohen's d
    pooled_std = np.sqrt((interested.std()**2 + not_interested.std()**2) / 2)
    cohens_d = (interested.mean() - not_interested.mean()) / pooled_std
    print(f"Cohen's d = {cohens_d:.3f}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot
    df_clean.boxplot(column=var_name, by='ai_label', ax=ax1)
    ax1.set_title(f'{var_label} by Interest Level')
    ax1.set_xlabel('Interest Level')
    ax1.set_ylabel(var_label)
    
    # Violin plot
    sns.violinplot(data=df_clean, x='ai_label', y=var_name, ax=ax2)
    ax2.set_title(f'{var_label} Distribution by Interest')
    ax2.set_xlabel('Interest Level')
    ax2.set_ylabel(var_label)
    
    plt.tight_layout()
    plt.savefig(f'results/study2/demographic_{var_name}_interest.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return t_stat, p_value

def main():
    """Main analysis function"""
    print("📊 Demographic Analysis of LLM Interest Labels")
    print("=" * 70)
    
    # Create output directory
    Path('results/study2').mkdir(parents=True, exist_ok=True)
    
    # Load and merge data
    df = load_and_merge_data()
    
    # Show basic distribution
    print("\nInterest Distribution in Merged Data:")
    print(df['ai_label'].value_counts())
    print(df['ai_label'].value_counts(normalize=True) * 100)
    
    # Define variables to analyze
    categorical_vars = [
        ('Gener Identity', 'Gender Identity'),
        ('Race', 'Race/Ethnicity'),
        ('Year_in_school', 'Year in School'),
        ('Parent_highest_level_education', 'Parent Education'),
        ('Current_employement', 'Employment Status'),
        ('Area_grew_up', 'Area Grew Up'),
        ('Substance_use_treatment', 'Substance Use Treatment'),
        ('Family_friend_substance_use_treatment', 'Family/Friend SU Treatment'),
        ('Mental_health_treatment', 'Mental Health Treatment'),
        ('Religious_affiliation', 'Religious Affiliation')
    ]
    
    continuous_vars = [
        ('Age', 'Age'),
        ('Household_income', 'Household Income'),
        ('Personal_Income', 'Personal Income'),
        ('Number_live_with', 'Number Living With'),
        ('Physical_safety_current_residence', 'Physical Safety Rating'),
        ('Safety_area_grew_up', 'Safety of Area Grew Up'),
        ('Frequency_talk_to_close_connections', 'Frequency Talk to Close Connections')
    ]
    
    # Analyze categorical variables
    print("\n" + "="*70)
    print("CATEGORICAL VARIABLE ANALYSIS")
    print("="*70)
    
    cat_results = []
    for var_name, var_label in categorical_vars:
        if var_name in df.columns:
            try:
                chi2, p_value = explore_categorical_relationships(df, var_name, var_label)
                cat_results.append({
                    'Variable': var_label,
                    'Chi-square': chi2,
                    'p-value': p_value,
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })
            except Exception as e:
                print(f"Error analyzing {var_name}: {e}")
    
    # Analyze continuous variables
    print("\n" + "="*70)
    print("CONTINUOUS VARIABLE ANALYSIS")
    print("="*70)
    
    cont_results = []
    for var_name, var_label in continuous_vars:
        if var_name in df.columns:
            try:
                t_stat, p_value = explore_continuous_relationships(df, var_name, var_label)
                cont_results.append({
                    'Variable': var_label,
                    't-statistic': t_stat,
                    'p-value': p_value,
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })
            except Exception as e:
                print(f"Error analyzing {var_name}: {e}")
    
    # Summary tables
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    
    if cat_results:
        cat_results_df = pd.DataFrame(cat_results)
        print("\nCategorical Variables:")
        print(cat_results_df.to_string(index=False))
        cat_results_df.to_csv('results/study2/demographic_categorical_results.csv', index=False)
    
    if cont_results:
        cont_results_df = pd.DataFrame(cont_results)
        print("\nContinuous Variables:")
        print(cont_results_df.to_string(index=False))
        cont_results_df.to_csv('results/study2/demographic_continuous_results.csv', index=False)
    
    # Create overall summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Interest distribution
    interest_counts = df['ai_label'].value_counts()
    ax.bar(interest_counts.index, interest_counts.values)
    ax.set_title('Distribution of Interest in SUD Counseling\n(Participants with Demographic Data)', fontsize=14)
    ax.set_xlabel('Interest Level', fontsize=12)
    ax.set_ylabel('Number of Participants', fontsize=12)
    
    # Add counts on bars
    for i, (label, count) in enumerate(interest_counts.items()):
        ax.text(i, count + 0.5, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/study2/demographic_interest_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save merged dataset
    df.to_csv('results/study2/merged_demographics_interest.csv', index=False)
    print(f"\n✅ Analysis complete! Merged dataset saved to: results/study2/merged_demographics_interest.csv")
    
    # Report missing participants
    llm_df = pd.read_csv('results/study2/clean_participant_dataset_fewshot.csv')
    missing_ids = set(llm_df['participant_id']) - set(df['participant_id'])
    if missing_ids:
        print(f"\n⚠️  Warning: {len(missing_ids)} participants have LLM labels but no demographic data:")
        print(f"   Missing IDs: {sorted(missing_ids)}")

if __name__ == "__main__":
    main()