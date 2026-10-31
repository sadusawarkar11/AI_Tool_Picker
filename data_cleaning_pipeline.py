"""
Professional Data Cleaning Pipeline for ToolMatcher AI
Fixes: Duplicate tools, inconsistent expertise levels, data quality issues
Author: Production ML Engineer
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re

class DataCleaningPipeline:
    """Professional data cleaning pipeline with validation and reporting"""
    
    def __init__(self, input_file='toolmatcher_ai_raw_dataset_v2.csv'):
        self.input_file = input_file
        self.df = None
        self.cleaning_report = {
            'original_records': 0,
            'cleaned_records': 0,
            'duplicates_removed': 0,
            'expertise_standardized': 0,
            'category_standardized': 0,
            'missing_values_handled': 0,
            'outliers_removed': 0
        }
    
    def load_data(self):
        """Load raw dataset"""
        print(f"\n{'='*70}")
        print("STEP 1: LOADING RAW DATA")
        print(f"{'='*70}")
        
        self.df = pd.read_csv(self.input_file)
        self.cleaning_report['original_records'] = len(self.df)
        
        print(f"✓ Loaded {len(self.df)} records")
        print(f"✓ Columns: {len(self.df.columns)}")
        print(f"✓ Unique tools: {self.df['tool_name'].nunique()}")
        
        return self
    
    def standardize_expertise_levels(self):
        """Standardize all expertise level variations"""
        print(f"\n{'='*70}")
        print("STEP 2: STANDARDIZING EXPERTISE LEVELS")
        print(f"{'='*70}")
        
        # Show current variations
        current_levels = self.df['expertise_level'].unique()
        print(f"\n⚠️  Found {len(current_levels)} variations:")
        for level in sorted(current_levels):
            count = len(self.df[self.df['expertise_level'] == level])
            print(f"   - '{level}' : {count} records")
        
        # Standardization mapping
        expertise_mapping = {
            # Beginner variations
            'BEGINNER': 'Beginner',
            'beginner': 'Beginner',
            'Beg.': 'Beginner',
            'beg': 'Beginner',
            'BEG': 'Beginner',
            
            # Intermediate variations
            'INTERMEDIATE': 'Intermediate',
            'intermediate': 'Intermediate',
            'Int.': 'Intermediate',
            'int': 'Intermediate',
            'INT': 'Intermediate',
            
            # Advanced variations
            'ADVANCED': 'Advanced',
            'advanced': 'Advanced',
            'Adv.': 'Advanced',
            'adv': 'Advanced',
            'ADV': 'Advanced'
        }
        
        # Apply standardization
        before_count = self.df['expertise_level'].nunique()
        self.df['expertise_level'] = self.df['expertise_level'].replace(expertise_mapping)
        after_count = self.df['expertise_level'].nunique()
        
        self.cleaning_report['expertise_standardized'] = before_count - after_count
        
        print(f"\n✅ STANDARDIZATION COMPLETE")
        print(f"   Before: {before_count} variations")
        print(f"   After: {after_count} standard levels")
        print(f"\n   Final levels:")
        for level in sorted(self.df['expertise_level'].unique()):
            count = len(self.df[self.df['expertise_level'] == level])
            print(f"   ✓ {level}: {count} records")
        
        return self
    
    def standardize_tool_names(self):
        """Standardize tool names - CRITICAL for deduplication"""
        print(f"\n{'='*70}")
        print("STEP 3: STANDARDIZING TOOL NAMES")
        print(f"{'='*70}")
        
        before_count = self.df['tool_name'].nunique()
        
        # First, clean whitespace and normalize case
        self.df['tool_name'] = self.df['tool_name'].str.strip()
        
        # Create comprehensive mapping for ALL variations
        # Strategy: convert to Title Case first, then apply specific corrections
        def standardize_name(name):
            if pd.isna(name):
                return name
            
            # Remove extra spaces
            name = ' '.join(name.split())
            
            # Known mappings (case-insensitive)
            name_lower = name.lower()
            
            standard_names = {
                'chatgpt': 'ChatGPT',
                'chat gpt': 'ChatGPT',
                'midjourney': 'Midjourney',
                'leonardo ai': 'Leonardo AI',
                'leonardo.ai': 'Leonardo AI',
                'claude': 'Claude',
                'claude ai': 'Claude',
                'cursor': 'Cursor',
                'gemini': 'Gemini',
                'google gemini': 'Gemini',
                'perplexity': 'Perplexity',
                'perplexity ai': 'Perplexity',
                'github copilot': 'GitHub Copilot',
                'ideogram': 'Ideogram',
                'canva ai': 'Canva AI',
                'canva': 'Canva AI',
                'adobe firefly': 'Adobe Firefly',
                'firefly': 'Adobe Firefly',
                'deepseek': 'DeepSeek',
                'deep seek': 'DeepSeek',
                'otter.ai': 'Otter.ai',
                'otterai': 'Otter.ai',
                'elevenlabs': 'ElevenLabs',
                'eleven labs': 'ElevenLabs',
                'runway': 'Runway',
                'runway ml': 'Runway',
                'runwayml': 'Runway',
                'synthesia': 'Synthesia',
                'heygen': 'HeyGen',
                'hey gen': 'HeyGen',
                'descript': 'Descript',
                'fireflies.ai': 'Fireflies.ai',
                'fireflies': 'Fireflies.ai',
                'notion ai': 'Notion AI',
                'notion': 'Notion AI',
                'grammarly': 'Grammarly',
                'quilbot': 'QuillBot',
                'quillbot': 'QuillBot',
                'jasper': 'Jasper',
                'jasper ai': 'Jasper',
                'copy.ai': 'Copy.ai',
                'copyai': 'Copy.ai',
                'writesonic': 'Writesonic',
                'dall-e': 'DALL-E',
                'dalle': 'DALL-E',
                'dall-e 3': 'DALL-E',
                'stable diffusion': 'Stable Diffusion',
                'microsoft copilot': 'Microsoft Copilot',
                'replit': 'Replit',
                'bolt.new': 'Bolt.new',
                'bolt': 'Bolt.new',
                'v0': 'v0',
                'lovable': 'Lovable',
                'windsurf': 'Windsurf',
                'amazon q developer': 'Amazon Q Developer',
                'amazon q': 'Amazon Q Developer',
                'elicit': 'Elicit',
                'consensus': 'Consensus',
                'notebooklm': 'NotebookLM',
                'notebook lm': 'NotebookLM',
                'deepl': 'DeepL',
                'deepl translator': 'DeepL',
                'mem': 'Mem',
                'mem ai': 'Mem',
                'taskade': 'Taskade',
                'taskade ai': 'Taskade',
                'notta': 'Notta',
                'notta ai': 'Notta',
                'fathom': 'Fathom',
                'remove.bg': 'Remove.bg',
                'removebg': 'Remove.bg',
                'photoroom': 'Photoroom',
                'photoroom ai': 'Photoroom',
                'kling ai': 'Kling AI',
                'klingai': 'Kling AI',
                'luma': 'Luma',
                'luma dream machine': 'Luma',
                'murf ai': 'Murf AI',
                'murf': 'Murf AI',
                'aiva': 'AIVA',
                'soundraw': 'Soundraw',
                'suno': 'Suno',
                'suno ai': 'Suno',
                'udio': 'Udio',
                'prezi ai': 'Prezi AI',
                'prezi': 'Prezi AI',
                'gamma': 'Gamma',
                'gamma ai': 'Gamma',
                'beautiful.ai': 'Beautiful.ai',
                'beautifulai': 'Beautiful.ai',
                'microsoft translator': 'Microsoft Translator',
                'mistral le chat': 'Mistral Le Chat',
                'mistral': 'Mistral Le Chat',
            }
            
            if name_lower in standard_names:
                return standard_names[name_lower]
            
            # If not in mapping, return cleaned Title Case version
            return name.title()
        
        self.df['tool_name'] = self.df['tool_name'].apply(standardize_name)
        
        after_count = self.df['tool_name'].nunique()
        tools_merged = before_count - after_count
        
        print(f"\n✅ TOOL NAME STANDARDIZATION COMPLETE")
        print(f"   Before: {before_count} tool names")
        print(f"   After: {after_count} tool names")
        print(f"   Variations merged: {tools_merged}")
        
        # Show top 20 tools
        print(f"\n   Top 20 tools after standardization:")
        for tool, count in self.df['tool_name'].value_counts().head(20).items():
            print(f"   - {tool}: {count} records")
        
        return self
    
    def standardize_task_categories(self):
        """Standardize task category naming"""
        print(f"\n{'='*70}")
        print("STEP 4: STANDARDIZING TASK CATEGORIES")
        print(f"{'='*70}")
        
        # Normalize case variations
        category_mapping = {
            'general assistant': 'General Assistant',
            'GENERAL ASSISTANT': 'General Assistant',
            'general chat': 'General Chat',
            'GENERAL CHAT': 'General Chat',
            'coding': 'Coding',
            'CODING': 'Coding',
            'writing & marketing': 'Writing & Marketing',
            'WRITING & MARKETING': 'Writing & Marketing',
            'voice & audio': 'Voice & Audio',
            'VOICE & AUDIO': 'Voice & Audio',
            'Translate': 'Translation',
            'translate': 'Translation',
            'Image Gen': 'Image Generation',
            'image gen': 'Image Generation',
            'Data Analysis': 'Data Analytics',
        }
        
        before_count = self.df['task_category'].nunique()
        self.df['task_category'] = self.df['task_category'].replace(category_mapping)
        after_count = self.df['task_category'].nunique()
        
        self.cleaning_report['category_standardized'] = before_count - after_count
        
        print(f"✓ Standardized from {before_count} to {after_count} categories")
        print(f"\n   Top 10 categories:")
        for cat, count in self.df['task_category'].value_counts().head(10).items():
            print(f"   - {cat}: {count} records")
        
        return self
    
    def remove_duplicates(self):
        """Remove duplicate records intelligently"""
        print(f"\n{'='*70}")
        print("STEP 5: REMOVING DUPLICATES")
        print(f"{'='*70}")
        
        before_count = len(self.df)
        
        # Check for exact duplicates
        exact_dupes = self.df.duplicated().sum()
        print(f"\n   Exact duplicates: {exact_dupes}")
        
        # Remove exact duplicates
        self.df = self.df.drop_duplicates()
        
        # Remove near-duplicates (same tool, expertise, category)
        # Keep the record with highest efficiency_rate_pct
        print(f"\n   Removing near-duplicates (same tool + expertise + category)...")
        
        # Sort by efficiency to keep best records
        self.df = self.df.sort_values('efficiency_rate_pct', ascending=False)
        
        # Keep first occurrence (highest efficiency) for each combination
        self.df = self.df.drop_duplicates(
            subset=['tool_name', 'expertise_level', 'task_category'],
            keep='first'
        )
        
        after_count = len(self.df)
        duplicates_removed = before_count - after_count
        self.cleaning_report['duplicates_removed'] = duplicates_removed
        
        print(f"\n✅ DEDUPLICATION COMPLETE")
        print(f"   Records before: {before_count}")
        print(f"   Records after: {after_count}")
        print(f"   Duplicates removed: {duplicates_removed}")
        
        return self
    
    def clean_numeric_columns(self):
        """Clean and convert numeric columns with mixed formats"""
        print(f"\n{'='*70}")
        print("STEP 6: CLEANING NUMERIC COLUMNS")
        print(f"{'='*70}")
        
        # Fix efficiency_rate_pct (remove %, spaces, convert to float)
        if 'efficiency_rate_pct' in self.df.columns:
            self.df['efficiency_rate_pct'] = (
                self.df['efficiency_rate_pct']
                .astype(str)
                .str.replace('%', '')
                .str.replace(' ', '')
                .str.strip()
            )
            self.df['efficiency_rate_pct'] = pd.to_numeric(self.df['efficiency_rate_pct'], errors='coerce')
            print(f"   ✓ Cleaned efficiency_rate_pct column")
        
        # Clean price columns (remove $, commas, spaces)
        price_cols = ['monthly_price_usd', 'yearly_price_usd', 'monthly_budget_usd', 'standard_plan_monthly_usd']
        for col in price_cols:
            if col in self.df.columns:
                self.df[col] = (
                    self.df[col]
                    .astype(str)
                    .str.replace('$', '')
                    .str.replace(',', '')
                    .str.replace(' ', '')
                    .str.strip()
                )
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                print(f"   ✓ Cleaned {col} column")
        
        # Clean other numeric columns
        numeric_cols = ['output_quality_1_10', 'ease_of_use_1_5']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                print(f"   ✓ Cleaned {col} column")
        
        # Convert boolean columns
        bool_cols = ['free_plan_available', 'annual_billing_available']
        for col in bool_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].map({
                    'Yes': 1, 'No': 0, 'yes': 1, 'no': 0,
                    '1': 1, '0': 0, 1: 1, 0: 0, 
                    True: 1, False: 0, 'TRUE': 1, 'FALSE': 0
                }).fillna(0).astype(int)
                print(f"   ✓ Cleaned {col} column")
        
        return self
    
    def handle_missing_values(self):
        """Handle missing values professionally"""
        print(f"\n{'='*70}")
        print("STEP 7: HANDLING MISSING VALUES")
        print(f"{'='*70}")
        
        missing_before = self.df.isnull().sum().sum()
        
        # Fill numeric columns with median
        numeric_cols = ['efficiency_rate_pct', 'output_quality_1_10', 'ease_of_use_1_5']
        for col in numeric_cols:
            if col in self.df.columns:
                median_val = self.df[col].median()
                missing = self.df[col].isnull().sum()
                if missing > 0:
                    self.df[col].fillna(median_val, inplace=True)
                    print(f"   ✓ Filled {missing} missing values in {col} with median: {median_val:.2f}")
        
        # Fill categorical with mode or 'Unknown'
        if self.df['platform'].isnull().sum() > 0:
            self.df['platform'].fillna('Web', inplace=True)
            print(f"   ✓ Filled missing platforms with 'Web'")
        
        if self.df['best_model'].isnull().sum() > 0:
            mode_model = self.df['best_model'].mode()[0] if len(self.df['best_model'].mode()) > 0 else 'GPT-4'
            self.df['best_model'].fillna(mode_model, inplace=True)
            print(f"   ✓ Filled missing models with '{mode_model}'")
        
        # Fill price columns
        if self.df['monthly_price_usd'].isnull().sum() > 0:
            self.df['monthly_price_usd'].fillna(0, inplace=True)
            print(f"   ✓ Filled missing monthly prices with 0 (free)")
        
        missing_after = self.df.isnull().sum().sum()
        self.cleaning_report['missing_values_handled'] = missing_before - missing_after
        
        print(f"\n✅ Missing values handled: {missing_before} → {missing_after}")
        
        return self
    
    def remove_outliers(self):
        """Remove statistical outliers"""
        print(f"\n{'='*70}")
        print("STEP 8: REMOVING OUTLIERS")
        print(f"{'='*70}")
        
        before_count = len(self.df)
        
        # Remove efficiency outliers (should be 0-100%)
        outliers_efficiency = (self.df['efficiency_rate_pct'] < 0) | (self.df['efficiency_rate_pct'] > 100)
        outliers_removed_eff = outliers_efficiency.sum()
        self.df = self.df[~outliers_efficiency]
        
        if outliers_removed_eff > 0:
            print(f"   ✓ Removed {outliers_removed_eff} efficiency outliers")
        
        # Remove quality outliers (should be 1-10)
        outliers_quality = (self.df['output_quality_1_10'] < 1) | (self.df['output_quality_1_10'] > 10)
        outliers_removed_qual = outliers_quality.sum()
        self.df = self.df[~outliers_quality]
        
        if outliers_removed_qual > 0:
            print(f"   ✓ Removed {outliers_removed_qual} quality outliers")
        
        # Remove ease outliers (should be 1-5)
        outliers_ease = (self.df['ease_of_use_1_5'] < 1) | (self.df['ease_of_use_1_5'] > 5)
        outliers_removed_ease = outliers_ease.sum()
        self.df = self.df[~outliers_ease]
        
        if outliers_removed_ease > 0:
            print(f"   ✓ Removed {outliers_removed_ease} ease-of-use outliers")
        
        after_count = len(self.df)
        self.cleaning_report['outliers_removed'] = before_count - after_count
        
        print(f"\n✅ Total outliers removed: {before_count - after_count}")
        
        return self
    
    def validate_data_quality(self):
        """Final validation checks"""
        print(f"\n{'='*70}")
        print("STEP 9: DATA QUALITY VALIDATION")
        print(f"{'='*70}")
        
        # Check expertise levels
        valid_expertise = {'Beginner', 'Intermediate', 'Advanced'}
        invalid_expertise = set(self.df['expertise_level'].unique()) - valid_expertise
        
        if invalid_expertise:
            print(f"   ⚠️  Invalid expertise levels found: {invalid_expertise}")
        else:
            print(f"   ✓ All expertise levels valid: {valid_expertise}")
        
        # Check data ranges
        print(f"\n   Data Range Validation:")
        print(f"   ✓ Efficiency: {self.df['efficiency_rate_pct'].min():.1f}% - {self.df['efficiency_rate_pct'].max():.1f}%")
        print(f"   ✓ Quality: {self.df['output_quality_1_10'].min():.1f} - {self.df['output_quality_1_10'].max():.1f}")
        print(f"   ✓ Ease of use: {self.df['ease_of_use_1_5'].min():.1f} - {self.df['ease_of_use_1_5'].max():.1f}")
        
        # Check for duplicates
        dupes = self.df.duplicated(subset=['tool_name', 'expertise_level', 'task_category']).sum()
        if dupes > 0:
            print(f"   ⚠️  WARNING: {dupes} duplicate tool combinations still exist!")
        else:
            print(f"   ✓ No duplicate tool combinations")
        
        # Check unique tools
        unique_tools = self.df['tool_name'].nunique()
        print(f"   ✓ Unique tools: {unique_tools}")
        
        return self
    
    def save_cleaned_data(self, output_file='cleaned_toolmatcher_dataset.csv'):
        """Save cleaned dataset"""
        print(f"\n{'='*70}")
        print("STEP 10: SAVING CLEANED DATA")
        print(f"{'='*70}")
        
        self.cleaning_report['cleaned_records'] = len(self.df)
        
        # Save cleaned data
        self.df.to_csv(output_file, index=False)
        print(f"✓ Saved {len(self.df)} records to {output_file}")
        
        # Generate cleaning report
        report_file = 'data_cleaning_report.txt'
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DATA CLEANING REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            f.write("SUMMARY:\n")
            f.write(f"  Original records: {self.cleaning_report['original_records']}\n")
            f.write(f"  Cleaned records: {self.cleaning_report['cleaned_records']}\n")
            f.write(f"  Records removed: {self.cleaning_report['original_records'] - self.cleaning_report['cleaned_records']}\n")
            retained_pct = self.cleaning_report['cleaned_records'] / self.cleaning_report['original_records'] * 100
            f.write(f"  Rows retained: {retained_pct:.1f}% of the original dataset\n\n")
            
            f.write("CLEANING OPERATIONS:\n")
            f.write(f"  Expertise levels standardized: {self.cleaning_report['expertise_standardized']} variations\n")
            f.write(f"  Categories standardized: {self.cleaning_report['category_standardized']} variations\n")
            f.write(f"  Duplicates removed: {self.cleaning_report['duplicates_removed']}\n")
            f.write(f"  Missing values handled: {self.cleaning_report['missing_values_handled']}\n")
            f.write(f"  Outliers removed: {self.cleaning_report['outliers_removed']}\n\n")
            
            f.write("FINAL DATA QUALITY:\n")
            f.write(f"  Unique tools: {self.df['tool_name'].nunique()}\n")
            f.write(f"  Expertise levels: {', '.join(sorted(self.df['expertise_level'].unique()))}\n")
            f.write(f"  Task categories: {self.df['task_category'].nunique()}\n")
            f.write(f"  Remaining missing cells: {self.df.isnull().sum().sum()}\n")
            f.write(f"  Duplicates: {self.df.duplicated(subset=['tool_name', 'expertise_level', 'task_category']).sum()}\n")
            f.write("="*70 + "\n")
        
        print(f"✓ Saved cleaning report to {report_file}")
        
        return self
    
    def print_summary(self):
        """Print final summary"""
        print(f"\n{'='*70}")
        print("🎯 DATA CLEANING PIPELINE COMPLETE!")
        print(f"{'='*70}")
        
        print(f"\n📊 BEFORE → AFTER:")
        print(f"   Records: {self.cleaning_report['original_records']} → {self.cleaning_report['cleaned_records']}")
        print(f"   Removed: {self.cleaning_report['original_records'] - self.cleaning_report['cleaned_records']}")
        print(f"   Rows retained: {(self.cleaning_report['cleaned_records']/self.cleaning_report['original_records']*100):.1f}%")
        
        print(f"\n✅ CLEANING OPERATIONS:")
        print(f"   ✓ Expertise levels standardized: {self.cleaning_report['expertise_standardized']} variations")
        print(f"   ✓ Categories standardized: {self.cleaning_report['category_standardized']} variations")
        print(f"   ✓ Duplicates removed: {self.cleaning_report['duplicates_removed']}")
        print(f"   ✓ Missing values handled: {self.cleaning_report['missing_values_handled']}")
        print(f"   ✓ Outliers removed: {self.cleaning_report['outliers_removed']}")
        
        print(f"\n🎖️  PRODUCTION READY!")
        print(f"   ✓ {self.df['tool_name'].nunique()} unique tools")
        print(f"   ✓ {len(self.df['expertise_level'].unique())} expertise levels")
        print(f"   ✓ {self.df['task_category'].nunique()} task categories")
        print(f"   ✓ 0 duplicate combinations")
        print(f"   ✓ Clean, validated, professional dataset")
        
        print(f"\n{'='*70}\n")
    
    def run_pipeline(self, output_file='cleaned_toolmatcher_dataset.csv'):
        """Run complete cleaning pipeline"""
        (self
         .load_data()
         .standardize_expertise_levels()
         .standardize_tool_names()
         .standardize_task_categories()
         .remove_duplicates()
         .clean_numeric_columns()
         .handle_missing_values()
         .remove_outliers()
         .validate_data_quality()
         .save_cleaned_data(output_file)
         .print_summary())
        
        return self.df


if __name__ == '__main__':
    # Run the professional cleaning pipeline
    pipeline = DataCleaningPipeline('toolmatcher_ai_raw_dataset_v2.csv')
    cleaned_df = pipeline.run_pipeline('cleaned_toolmatcher_dataset.csv')
    
    print("\n🚀 Ready to train models with clean data!")
    print("   Run: python advanced_model_training.py\n")
