#!/usr/bin/env python3
"""
Spot check SEC financial reports to compare tags used by different companies.

This script loads 10-Q reports from specified companies for a given quarter
and compares the XBRL tags they use, identifying common and unique tags.

Uses financial domain language models for semantic similarity matching.
"""

import pandas as pd
from utility_data import load_and_join_sec_xbrl_data, get_cik_ticker_mapping
from config import DATA_BASE_DIR
import os
import re
from typing import Set, Tuple, Dict, List, Optional
from difflib import SequenceMatcher
import numpy as np
import pickle
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Try to import sentence-transformers for financial domain semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    import torch
    SEMANTIC_MODEL_AVAILABLE = True
except ImportError:
    SEMANTIC_MODEL_AVAILABLE = False
    print("⚠️  Warning: sentence-transformers not available. Install with: pip install sentence-transformers torch")
    print("   Falling back to basic string matching.")

# Companies to check (ticker symbols) - diverse set from Fortune 500 across industries
COMPANIES_TO_CHECK = {
    # Technology
    'Intuit': 'INTU',
    'Apple': 'AAPL',
    'Microsoft': 'MSFT',
    'Tesla': 'TSLA',
    # Financial Services
    'JPMorgan Chase': 'JPM',
    # Retail
    'Walmart': 'WMT',
    'Amazon': 'AMZN',
    # Healthcare
    'Johnson & Johnson': 'JNJ',
    # Energy
    'Exxon Mobil': 'XOM',
    # Consumer Goods
    'Procter & Gamble': 'PG'
}

# Similarity threshold for considering a match valid (0.0 to 1.0)
SIMILARITY_THRESHOLD = 0.9

# Quarter directory to check (e.g., '2025q2')
QUARTER_DIR = '2025q2'

KEY_METRICS = [
    'AccountsReceivableNetCurrent',
    'Assets',
    'AssetsCurrent',
    'CashAndCashEquivalentsAtCarryingValue',
    'CommonStockSharesOutstanding',  # For per-share metrics
    'CostOfGoodsAndServicesSold',
    'DepreciationDepletionAndAmortization',
    'GrossProfit',
    'InterestExpense',
    'InventoryNet',
    'LiabilitiesCurrent',
    'LongTermDebtNoncurrent',
    'NetCashProvidedByUsedInOperatingActivities',
    'NetIncomeLoss',
    'OperatingIncomeLoss',
    'PaymentsToAcquirePropertyPlantAndEquipment',
    'Revenues',
    'RevenueFromContractWithCustomerExcludingAssessedTax',   
    'StockholdersEquity'
]


PICKLE_FILE = 'metric_all_matches.pkl'

# =============================================================================
# FINANCIAL LANGUAGE MODEL FOR SEMANTIC SIMILARITY
# =============================================================================

class FinancialSemanticMatcher:
    """
    Uses financial domain language models for semantic similarity matching.
    
    Handles:
    - Financial terminology synonyms (sales/revenue, etc.)
    - Singular/plural forms
    - Compound words (long-term, cash flow, etc.)
    - Antonyms (long-term vs short-term)
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the semantic matcher with a financial domain sentence embedding model.
        
        Prioritizes financial-domain models that are fine-tuned on financial text:
        - Financial sentence transformers trained on SEC filings, financial reports, etc.
        - Falls back to general-purpose models if financial models unavailable
        
        Args:
            model_name: HuggingFace model name. If None, tries financial models first:
                       - 'willy-arison/bge-base-financial-willy3' (financial domain, best)
                       - '1tunadogan/gte-small-financial-matryoshka' (financial domain, smaller)
                       - 'sentence-transformers/all-MiniLM-L6-v2' (general purpose fallback)
        """
        self.model = None
        self.use_semantic = False
        
        if not SEMANTIC_MODEL_AVAILABLE:
            print("⚠️  Using fallback string matching (install sentence-transformers for better results)")
            return
        
        try:
            # Prioritize financial-domain sentence transformers
            # These are fine-tuned on financial text (SEC filings, reports, etc.)
            financial_models = [
                'willy-arison/bge-base-financial-willy3',      # Financial domain, BGE base
                '1tunadogan/gte-small-financial-matryoshka',   # Financial domain, smaller/faster
            ]
            
            # Fallback to general-purpose models if financial models unavailable
            general_models = [
                'sentence-transformers/all-MiniLM-L6-v2',  # Fast, good quality
                'sentence-transformers/all-mpnet-base-v2',  # Better quality, slower
            ]
            
            all_models = financial_models + general_models
            
            if model_name:
                all_models.insert(0, model_name)
            
            for model_to_try in all_models:
                try:
                    model_type = "🟢 Financial domain" if model_to_try in financial_models else "🔵 General purpose"
                    print(f"🔄 Loading {model_type} model: {model_to_try}...")
                    self.model = SentenceTransformer(model_to_try)
                    self.use_semantic = True
                    print(f"✅ Loaded {model_type} model: {model_to_try}")
                    break
                except Exception as e:
                    print(f"⚠️  Failed to load {model_to_try}: {e}")
                    continue
            
            if not self.use_semantic:
                print("⚠️  Could not load any sentence transformer model, using fallback matching")
        except Exception as e:
            print(f"⚠️  Error initializing semantic model: {e}")
            print("   Using fallback string matching")
    
    def compute_phrase_level_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two complete phrases/texts.
        
        This method treats each input as a whole phrase and computes similarity
        based on the full semantic meaning, not individual words.
        
        Args:
            text1: First text/phrase (e.g., "Net Income")
            text2: Second text/phrase (e.g., "NetIncomeLoss")
            
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        if not self.use_semantic or self.model is None:
            # Fallback to string similarity
            return self._fallback_similarity(text1, text2)
        
        try:
            # Get embeddings for both texts
            embeddings = self.model.encode([text1, text2], convert_to_tensor=True)
            
            # Compute cosine similarity
            from torch.nn.functional import cosine_similarity
            similarity = cosine_similarity(embeddings[0:1], embeddings[1:2]).item()
            
            # Normalize to 0-1 range (cosine similarity is -1 to 1)
            return max(0.0, (similarity + 1) / 2)
        except Exception as e:
            print(f"⚠️  Error computing semantic similarity: {e}")
            return self._fallback_similarity(text1, text2)
    
    def compute_similarity_batch(self, texts1: List[str], texts2: List[str]) -> np.ndarray:
        """
        Compute batch phrase-level similarity between two lists of texts efficiently.
        
        This is a batch version of compute_phrase_level_similarity() for processing
        multiple text pairs at once, which is more efficient than calling the method
        multiple times.
        
        Args:
            texts1: List of first texts/phrases
            texts2: List of second texts/phrases (must be same length as texts1)
            
        Returns:
            numpy array of similarity scores (phrase-level)
        """
        if not self.use_semantic or self.model is None:
            # Fallback: compute pairwise
            return np.array([self._fallback_similarity(t1, t2) for t1, t2 in zip(texts1, texts2)])
        
        try:
            # Batch encode all texts
            all_texts = list(set(texts1 + texts2))
            embeddings_dict = {}
            batch_embeddings = self.model.encode(all_texts, convert_to_tensor=True, show_progress_bar=False)
            
            for text, emb in zip(all_texts, batch_embeddings):
                embeddings_dict[text] = emb
            
            # Compute cosine similarities
            from torch.nn.functional import cosine_similarity
            similarities = []
            for t1, t2 in zip(texts1, texts2):
                emb1 = embeddings_dict[t1].unsqueeze(0)
                emb2 = embeddings_dict[t2].unsqueeze(0)
                sim = cosine_similarity(emb1, emb2).item()
                # Normalize to 0-1 range
                similarities.append(max(0.0, (sim + 1) / 2))
            
            return np.array(similarities)
        except Exception as e:
            # Fallback on error
            return np.array([self._fallback_similarity(t1, t2) for t1, t2 in zip(texts1, texts2)])
    
    def compute_word_level_similarity(self, words1: List[str], words2: List[str]) -> float:
        """
        Compute similarity between two lists of words using word-level semantic matching.
        
        This method compares individual words between the two lists, finding the best
        match for each word, then combines the scores. This is useful for handling
        word order differences (e.g., "Current Assets" vs "AssetsCurrent").
        
        Args:
            words1: List of words from first phrase
            words2: List of words from second phrase
            
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        if not words1 or not words2:
            return 0.0
        
        # Compute pairwise similarities between all words
        word_scores = []
        for word1 in words1:
            best_score = 0.0
            for word2 in words2:
                # Compare individual words using phrase-level similarity
                score = self.compute_phrase_level_similarity(word1, word2)
                best_score = max(best_score, score)
            word_scores.append(best_score)
        
        # Average word match score
        avg_score = sum(word_scores) / len(word_scores)
        
        # Coverage: how many words matched well
        good_matches = sum(1 for s in word_scores if s >= 0.7)
        coverage = good_matches / len(words1)
        
        # Combined score
        final_score = (avg_score * 0.6) + (coverage * 0.4)
        
        return min(final_score, 1.0)
    
    def _fallback_similarity(self, text1: str, text2: str) -> float:
        """Fallback string similarity when semantic model is unavailable."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


# Global semantic matcher instance (lazy initialization)
_semantic_matcher: Optional[FinancialSemanticMatcher] = None

def get_semantic_matcher() -> FinancialSemanticMatcher:
    """Get or create the global semantic matcher instance."""
    global _semantic_matcher
    if _semantic_matcher is None:
        _semantic_matcher = FinancialSemanticMatcher()
    return _semantic_matcher

# =============================================================================
# TEXT PROCESSING UTILITIES
# =============================================================================

def split_camel_case(text):
    """
    Split camelCase or PascalCase string into words.
    
    Examples:
        "OtherNoncashIncomeExpense" -> ["Other", "Noncash", "Income", "Expense"]
        "NetIncomeLoss" -> ["Net", "Income", "Loss"]
        "OperatingCashFlow" -> ["Operating", "Cash", "Flow"]
    
    Args:
        text (str): camelCase or PascalCase string
        
    Returns:
        list: List of words
    """
    # Insert space before capital letters, then split
    words = re.sub(r'(?<!^)(?=[A-Z])', ' ', text).split()
    return words

def extract_words(text):
    """
    Extract and normalize words from a text string, handling both camelCase and space-separated formats.
    
    This function:
    1. Extracts words by splitting on spaces or camelCase boundaries
    2. Handles hyphenated words (e.g., "Long-Term" -> ["Long", "Term"])
    3. Normalizes each word (lowercase, remove punctuation, remove stop words)
    
    Args:
        text (str): Text to extract words from (e.g., "Net Income", "NetIncomeLoss", "Long-Term Investment")
        
    Returns:
        list: List of normalized words
    """
    # Stop words to remove during normalization
    stop_words = ['and', 'of', 'the', 'in', 'at', 'on', 'to', 'from', 'by', 'with']
    
    # First, try splitting by spaces (for key metrics like "Net Income")
    words = text.split()
    
    # If no spaces, try camelCase splitting (for tags like "NetIncomeLoss")
    if len(words) == 1:
        words = split_camel_case(text)
    
    # Split hyphenated words (e.g., "Long-Term" -> ["Long", "Term"])
    expanded_words = []
    for word in words:
        # Split on hyphens
        hyphen_split = word.split('-')
        expanded_words.extend(hyphen_split)
    
    # Normalize all words: lowercase, remove punctuation, remove stop words
    normalized_words = []
    for word in expanded_words:
        if len(word) == 0:
            continue
        
        # Convert to lowercase
        normalized = word.lower()
        # Remove hyphens and other punctuation
        normalized = re.sub(r'[^\w]', '', normalized)
        # Remove stop words
        for stop_word in stop_words:
            normalized = normalized.replace(stop_word, '')
        
        # Only add non-empty normalized words
        if len(normalized) > 0:
            normalized_words.append(normalized)
    
    return normalized_words

# =============================================================================
# SIMILARITY FUNCTIONS (USING SEMANTIC MODEL)
# =============================================================================

def similarity_score(word1: str, word2: str) -> float:
    """
    Calculate semantic similarity score between two words using financial domain model.
    
    Args:
        word1 (str): First word
        word2 (str): Second word
        
    Returns:
        float: Similarity score between 0.0 and 1.0
    """
    matcher = get_semantic_matcher()
    return matcher.compute_phrase_level_similarity(word1, word2)

def word_similarity(key_metric_words: list, tag_words: list) -> float:
    """
    Calculate similarity score between two lists of words using semantic matching.
    
    Args:
        key_metric_words (list): List of normalized words from key metric
        tag_words (list): List of normalized words from tag
        
    Returns:
        float: Similarity score between 0.0 and 1.0
    """
    if not key_metric_words or not tag_words:
        return 0.0
    
    matcher = get_semantic_matcher()
    return matcher.compute_word_level_similarity(key_metric_words, tag_words)

def _fast_prefilter_tags(key_metric: str, available_tags: Set[str], top_k: int = 50) -> List[str]:
    """
    Fast pre-filtering using string similarity to reduce candidates before expensive semantic matching.
    
    Args:
        key_metric: Key metric to match
        available_tags: All available tags
        top_k: Number of top candidates to return
        
    Returns:
        List of top candidate tags for semantic matching
    """
    key_words = set(word.lower() for word in extract_words(key_metric))
    if not key_words:
        return list(available_tags)[:top_k]
    
    # Fast string-based scoring
    candidates = []
    for tag in available_tags:
        tag_words = set(word.lower() for word in extract_words(tag))
        
        # Fast word overlap score
        overlap = len(key_words & tag_words) / max(len(key_words), 1)
        
        # Fast string similarity
        string_sim = SequenceMatcher(None, key_metric.lower(), tag.lower()).ratio()
        
        # Combined fast score
        fast_score = (overlap * 0.6) + (string_sim * 0.4)
        candidates.append((tag, fast_score))
    
    # Sort by fast score and return top K
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [tag for tag, _ in candidates[:top_k]]

def find_tag_matches(key_metric, available_tags, threshold=SIMILARITY_THRESHOLD):
    """
    Find the closest matching tag(s) for a key metric using semantic word-level matching.
    
    This function uses financial domain language models for semantic understanding:
    - Handles synonyms automatically (e.g., "sales" ≈ "revenue")
    - Understands compound words (e.g., "longterm" ≈ "long term")
    - Detects semantic relationships beyond string similarity
    
    Optimized with fast pre-filtering to reduce computation time.
    
    Args:
        key_metric (str): The key metric name to search for (e.g., "Operating Cash Flow")
        available_tags (set): Set of available tags in the 10-Q report (e.g., {"OperatingCashFlow"})
        threshold (float): Minimum similarity threshold (default: 0.9)
        
    Returns:
        list: List of tuples (tag_name, similarity_score) sorted by score (descending), 
              with shortest tag preferred when all words match legitimately
    """
    # Fast pre-filter to reduce candidates (much faster than semantic matching)
    candidate_tags = _fast_prefilter_tags(key_metric, available_tags, top_k=50)
    
    matches = []
    
    # Extract words from key metric (order-independent)
    key_metric_words = extract_words(key_metric)
    key_words_set = set(key_metric_words)
    
    if not key_metric_words:
        return []
    
    matcher = get_semantic_matcher()
    
    # Only process pre-filtered candidates
    for tag in candidate_tags:
        # Extract words from tag (handle camelCase)
        tag_words = extract_words(tag)
        
        if not tag_words:
            continue
        
        # Use sets for order-independent word comparison
        tag_words_set = set(tag_words)
        
        # Check if all key metric words are present in tag (order-independent using sets)
        all_words_present = key_words_set.issubset(tag_words_set) and len(key_words_set) > 0
        
        # Check for exact word match (all words match, order-independent)
        exact_word_match = key_words_set == tag_words_set
        
        if exact_word_match:
            combined_score = 1.0
        else:
            # Calculate word-level similarity using semantic model
            word_score = word_similarity(key_metric_words, tag_words)
            
            # Also check full phrase similarity
            phrase_score = matcher.compute_phrase_level_similarity(key_metric, tag)
            
            # Combined score: word-level matching is primary, phrase-level is secondary
            combined_score = (word_score * 0.7) + (phrase_score * 0.3)
            
            if all_words_present and combined_score >= 0.7:
                # All words present - boost the score
                combined_score = min(combined_score + 0.1, 1.0)
        
        if combined_score >= threshold:
            # Store tuple: (tag, score, all_words_present, tag_length)
            matches.append((tag, combined_score, all_words_present, len(tag)))
    
    # Sort matches with multi-level criteria:
    # 1. Primary: score (descending)
    # 2. Secondary: all words present (True first - prioritize complete matches)
    # 3. Tertiary: tag length (ascending - shortest wins when all words match)
    matches.sort(key=lambda x: (-x[1], -x[2], x[3]))
    
    # Return top 5 matches (without the helper fields)
    return [(tag, score) for tag, score, _, _ in matches[:5]]

# =============================================================================
# DATA LOADING AND ANALYSIS FUNCTIONS
# =============================================================================

def top1_match_agree_stats(metric_top1_matches: Dict[str, Dict[str, Tuple[str, float]]]):
    """
    Analyze agreement of top 1 matches across companies and categorize metrics.
    
    Categories:
    1. All companies use same tag
    2. Some companies missing, but all that have it use same tag
    3. Different tags across companies
    4. No match in any company
    
    Args:
        metric_top1_matches: Dictionary mapping company names to dictionaries of 
                            metric -> (tag, score) for top 1 matches
    """
    print("\n" + "="*80)
    print("CATEGORIZATION: Tag Consistency Analysis (Top 1 Matches)")
    print("="*80)
    
    category1 = []  # All companies use same tag
    category2 = []  # Some missing, but all that have it use same tag
    category3 = []  # Different tags across companies
    category4 = []  # No match in any company
    
    for metric in KEY_METRICS:
        tags_used = []
        for company_name in COMPANIES_TO_CHECK.keys():
            if company_name in metric_top1_matches:
                tag, _ = metric_top1_matches[company_name].get(metric, (None, 0.0)) 
                if tag:
                    tags_used.append(tag)
        
        if not tags_used:
            # Category 4: No valid matches found in any company
            category4.append(metric)
            continue
        
        unique_tags = set(tags_used)
        
        if len(unique_tags) == 1:
            # All companies with this metric use the same tag
            if len(tags_used) == len([c for c in COMPANIES_TO_CHECK.keys() if c in metric_top1_matches]):
                category1.append(metric)
            else:
                category2.append(metric)
        else:
            category3.append(metric)
    
    print(f"\n✅ Category 1 (All companies use same tag): {len(category1)} metrics")
    for metric in category1:
        # Find first company with this metric (already filtered by threshold)
        for company_name in COMPANIES_TO_CHECK.keys():
            if company_name in metric_top1_matches:
                tag, _ = metric_top1_matches[company_name].get(metric, (None, 0.0))
                if tag:
                    print(f"   - {metric}: {tag}")
                    break
    
    print(f"\n⚠️  Category 2 (Some missing, but consistent when present): {len(category2)} metrics")
    for metric in category2:
        # Find tag and identify which companies have it vs missing it
        companies_with_tag = []
        companies_missing_tag = []
        tag_name = None
        
        for company_name in COMPANIES_TO_CHECK.keys():
            if company_name in metric_top1_matches:
                tag, _ = metric_top1_matches[company_name].get(metric, (None, 0.0))
                if tag:
                    companies_with_tag.append(company_name)
                    if tag_name is None:
                        tag_name = tag  # All should have same tag in Category 2
                else:
                    companies_missing_tag.append(company_name)
            else:
                companies_missing_tag.append(company_name)
        
        # Print the metric with tag and list missing companies
        if tag_name:
            missing_str = f" (missing in: {', '.join(companies_missing_tag)})" if companies_missing_tag else ""
            print(f"   - {metric}: {tag_name}{missing_str}")
    
    print(f"\n❌ Category 3 (Different tags across companies): {len(category3)} metrics")
    for metric in category3:
        print(f"   - {metric}:")
        for company_name in COMPANIES_TO_CHECK.keys():
            if company_name in metric_top1_matches:
                tag, score = metric_top1_matches[company_name].get(metric, (None, 0.0))
                # If tag exists, it's already filtered by threshold
                if tag:
                    print(f"     * {company_name}: {tag} ({score:.2f})")
    
    print(f"\n🚫 Category 4 (No match in any company): {len(category4)} metrics")
    for metric in category4:
        print(f"   - {metric}")

def find_most_common_tag_by_voting(metric_all_matches: Dict[str, Dict[str, List[Tuple[str, float]]]]):
    """
    Find the most common tag across companies by voting on all matches.
    
    For each metric, considers all matches (not just top 1) from all companies
    and finds the tag that appears most frequently across companies.
    
    Args:
        metric_all_matches: Dictionary mapping company names to dictionaries of
                           metric -> list of (tag, score) tuples for all matches
    """
    print("\n" + "="*80)
    print("VOTING ANALYSIS: Most Common Tag Across All Matches")
    print("="*80)
    
    # Get ticker mapping for display
    company_to_ticker = {name: ticker for name, ticker in COMPANIES_TO_CHECK.items()}
    most_common_tag_matched = {}

    for metric in KEY_METRICS:
        # Track which companies have each tag (each company votes once per tag)
        tag_to_companies = {}  # tag -> list of companies that have this tag in their matches
        
        for company_name in COMPANIES_TO_CHECK.keys():
            if company_name in metric_all_matches:
                matches = metric_all_matches[company_name].get(metric, [])
                # Extract all unique tags from this company's matches
                company_tags = set([tag for tag, score in matches])
                
                # Track which companies have each tag (company votes once per unique tag)
                for tag in company_tags:
                    if tag not in tag_to_companies:
                        tag_to_companies[tag] = []
                    if company_name not in tag_to_companies[tag]:
                        tag_to_companies[tag].append(company_name)
        
        if not tag_to_companies:
            continue
        
        # Sort tags by number of companies (descending) to find most common and runner-up
        sorted_tags = sorted(tag_to_companies.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Most common tag
        most_common_tag, companies_with_most_common = sorted_tags[0]
        company_tickers_most = [company_to_ticker.get(cn, cn) for cn in companies_with_most_common]
        
        print(f"\n{metric}:")
        print(f"  Most common tag: {most_common_tag}")
        print(f"  {len(companies_with_most_common)} companies use this tag: {', '.join(company_tickers_most)}")
        
        # Runner-up tag (if exists and different from most common)
        if len(sorted_tags) > 1:
            runner_up_tag, companies_with_runner_up = sorted_tags[1]
            company_tickers_runner_up = [company_to_ticker.get(cn, cn) for cn in companies_with_runner_up]
            print(f"  Runner-up tag: {runner_up_tag}")
            print(f"  {len(companies_with_runner_up)} companies use this tag: {', '.join(company_tickers_runner_up)}")

        most_common_tag_matched[metric] = most_common_tag

    return most_common_tag_matched

def tag_populated_stats(tags: List[str]) -> Dict[str, float]:
    """
    Calculate the percentage of companies whose 10-Q forms use specific tag(s).
    
    Args:
        tags (list): List of XBRL tags to check for (e.g., ['Revenues', 'Assets'])
    
    Returns:
        dict: Dictionary mapping each tag to its percentage of companies using it (0.0 to 100.0)
    """
    # Load quarterly dataset
    quarter_dir = os.path.join(DATA_BASE_DIR, QUARTER_DIR)
    df_joined = load_and_join_sec_xbrl_data([quarter_dir])
    
    # Filter for 10-Q forms only
    df_10q = df_joined[df_joined['form'] == '10-Q'].copy() 
    total_companies = df_10q['cik'].nunique()
    
    # Initialize result dictionary
    populated_percentages = {}
    if df_10q.empty or total_companies == 0:
        # Return 0.0 for all tags if no data
        for tag in tags:
            populated_percentages[tag] = 0.0
        return populated_percentages
    
    # Count distinct CIKs with 10-Q forms that use each tag
    for tag in tags:
        companies_with_tag = df_10q[df_10q['tag'] == tag]['cik'].nunique()
        percentage = (companies_with_tag / total_companies) * 100.0
        populated_percentages[tag] = percentage
    
    return populated_percentages

def find_10q_reports(quarter_dir, companies_cik_map):
    """
    Find 10-Q reports for specified companies in a given quarter directory.
    
    Args:
        quarter_dir (str): Path to quarter directory containing SEC data
        companies_cik_map (dict): Dictionary mapping company names to CIKs (as zero-padded strings)
        
    Returns:
        dict: Dictionary mapping company names to DataFrames with their 10-Q data
    """
    # Load SEC XBRL data for this quarter
    df_joined = load_and_join_sec_xbrl_data([quarter_dir])
    
    # Filter for 10-Q forms only
    df_10q = df_joined[df_joined['form'] == '10-Q'].copy()
    
    company_reports = {}
    for company_name, cik_str in companies_cik_map.items():
        # Convert CIK from zero-padded string (e.g., '0000896878') to integer (e.g., 896878)
        # SEC data stores CIK as integer, not zero-padded string
        try:
            cik_int = int(cik_str)
        except (ValueError, TypeError):
            print(f"⚠️  Invalid CIK format for {company_name}: {cik_str}")
            continue
        
        company_data = df_10q[df_10q['cik'] == cik_int].copy()
        if not company_data.empty:
            company_reports[company_name] = company_data
        else:
            print(f"⚠️  No 10-Q report found for {company_name} (CIK: {cik_int}) in {quarter_dir}")
    
    return company_reports

def main_compute_matches():
    """
    Compute all tag matches for key metrics across all companies.
    This is computationally expensive and may take hours to complete.
    
    Returns:
        tuple: (metric_top1_matches, metric_all_matches) dictionaries
    """
    # Get CIK mappings
    _, ticker_to_cik = get_cik_ticker_mapping()
    
    # Build company name to CIK mapping
    companies_cik_map = {}
    for company_name, ticker in COMPANIES_TO_CHECK.items():
        if ticker in ticker_to_cik:
            companies_cik_map[company_name] = ticker_to_cik[ticker]
        else:
            print(f"⚠️  Could not find CIK for ticker {ticker} ({company_name})")
    
    # Specify quarter to check
    quarter_dir = os.path.join(DATA_BASE_DIR, QUARTER_DIR)
    
    if not os.path.exists(quarter_dir):
        print(f"❌ Error: Quarter directory not found: {quarter_dir}")
        return None, None
    
    # Find 10-Q reports
    print("\n📄 Loading 10-Q reports...")
    company_reports = find_10q_reports(quarter_dir, companies_cik_map)
    
    if not company_reports:
        print("❌ No 10-Q reports found for any companies!")
        return None, None
    
    print(f"\n✅ Found 10-Q reports for {len(company_reports)} company/companies")
    
    # Initialize semantic matcher early
    print("\n🤖 Initializing semantic matcher...")
    get_semantic_matcher()

    # Check key metrics for each company
    print("\n🔍 Checking key metrics matches...")
    metric_top1_matches = {}  # Top 1 match (highest score) per company per metric
    metric_all_matches = {}    # All matches per company per metric
    
    for company_name, df_10q in company_reports.items():
        print(f"\n  Processing {company_name}...")
        
        if df_10q.empty:
            metric_top1_matches[company_name] = {}
            metric_all_matches[company_name] = {}
            continue
        
        # Get unique tags from this company's 10-Q
        available_tags = set(df_10q['tag'].unique())
        
        print(f"  🔍 Matching {len(KEY_METRICS)} metrics against {len(available_tags)} tags...")
        
        top1_matches = {}
        all_matches = {}
        for key_metric in tqdm(KEY_METRICS, desc=f"    Matching {company_name}", disable=not TQDM_AVAILABLE):
            matches = find_tag_matches(key_metric, available_tags)
            # Store all matches
            all_matches[key_metric] = matches  # List of tuples (tag, score)
            # Store top 1 match if available
            if matches:
                top1_matches[key_metric] = matches[0]  # (tag, score)
        
        metric_top1_matches[company_name] = top1_matches
        metric_all_matches[company_name] = all_matches
    
    # Save all matches to pickle file 
    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump({'metric_top1_matches': metric_top1_matches, 'metric_all_matches': metric_all_matches}, f)
    print(f"\n💾 Saved all matches to: {PICKLE_FILE}")
    
    return metric_top1_matches, metric_all_matches

def main():
    """
    Main function to spot check SEC reports and find tag matches for key metrics.
    Uses cached pickle file if available to avoid recomputing matches.
    """ 
    
    # Check if pickle file exists
    if os.path.exists(PICKLE_FILE):
        print(f"📂 Loading matches from cache: {PICKLE_FILE}")
        with open(PICKLE_FILE, 'rb') as f:
            cached_data = pickle.load(f)
            
            # Check if it's the new format (dict with both keys) or old format (just metric_all_matches)
            if isinstance(cached_data, dict) and 'metric_top1_matches' in cached_data and 'metric_all_matches' in cached_data:
                # New format
                metric_top1_matches = cached_data['metric_top1_matches']
                metric_all_matches = cached_data['metric_all_matches']
            else:
                # Old format: just metric_all_matches saved directly
                metric_all_matches = cached_data
                # Reconstruct top1 matches from all matches
                metric_top1_matches = {}
                for company_name, all_matches_dict in metric_all_matches.items():
                    top1_dict = {}
                    for metric, matches in all_matches_dict.items():
                        if matches:
                            top1_dict[metric] = matches[0]  # (tag, score)
                    metric_top1_matches[company_name] = top1_dict
        print("✅ Loaded cached matches successfully")
    else:
        print("🔄 No cache found. Computing matches (this may take hours)...")
        metric_top1_matches, metric_all_matches = main_compute_matches()
        if metric_top1_matches is None or metric_all_matches is None:
            print("❌ Failed to compute matches. Exiting.")
            return
        # Sav/update pickle file with all results (matching the read structure)
        with open(PICKLE_FILE, 'wb') as f:
            pickle.dump({
                'metric_top1_matches': metric_top1_matches,
                'metric_all_matches': metric_all_matches
            }, f)
        print(f"💾 Saved matches to: {PICKLE_FILE}")
        
    # Analyze top 1 match agreement across companies
    top1_match_agree_stats(metric_top1_matches)
    
    # Find most common tag by voting across all matches
    most_common_tag_matched = find_most_common_tag_by_voting(metric_all_matches)

    # Collect all tags from most common matches
    all_tags = list(most_common_tag_matched.values())
    tag_percentages = tag_populated_stats(all_tags)
    for metric, tag in most_common_tag_matched.items():
        percentage = tag_percentages.get(tag, 0.0)
        print(f"{metric}: {tag}  Percentage of companies: {percentage:.2f}%")

if __name__ == "__main__":
    main()



# # output trace, saved here for reference
# ================================================================================
# VOTING ANALYSIS: Most Common Tag Across All Matches
# ================================================================================

# AccountsReceivableNetCurrent:
#   Most common tag: AccountsReceivableNetCurrent
#   6 companies use this tag: AAPL, MSFT, TSLA, AMZN, JNJ, PG

# Assets:
#   Most common tag: Assets
#   10 companies use this tag: INTU, AAPL, MSFT, TSLA, JPM, WMT, AMZN, JNJ, XOM, PG
#   Runner-up tag: OtherAssetsNoncurrent
#   9 companies use this tag: INTU, AAPL, MSFT, TSLA, WMT, AMZN, JNJ, XOM, PG

# AssetsCurrent:
#   Most common tag: AssetsCurrent
#   9 companies use this tag: INTU, AAPL, MSFT, TSLA, WMT, AMZN, JNJ, XOM, PG
#   Runner-up tag: PrepaidExpenseAndOtherAssetsCurrent
#   5 companies use this tag: INTU, TSLA, WMT, JNJ, PG

# CashAndCashEquivalentsAtCarryingValue:
#   Most common tag: CashAndCashEquivalentsAtCarryingValue
#   8 companies use this tag: INTU, AAPL, MSFT, TSLA, WMT, AMZN, JNJ, XOM
#   Runner-up tag: RestrictedCashAndCashEquivalentsAtCarryingValue
#   1 companies use this tag: XOM

# CommonStockSharesOutstanding:
#   Most common tag: CommonStockSharesOutstanding
#   8 companies use this tag: INTU, AAPL, MSFT, TSLA, WMT, AMZN, XOM, PG

# CostOfGoodsAndServicesSold:
#   Most common tag: CostOfGoodsAndServicesSold
#   6 companies use this tag: INTU, AAPL, MSFT, AMZN, JNJ, PG
#   Runner-up tag: CostOfGoodsAndServicesSoldAmortization
#   1 companies use this tag: INTU

# DepreciationDepletionAndAmortization:
#   Most common tag: DepreciationDepletionAndAmortization
#   5 companies use this tag: AAPL, AMZN, JNJ, XOM, PG
#   Runner-up tag: AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment
#   2 companies use this tag: MSFT, JNJ

# GrossProfit:
#   Most common tag: GrossProfit
#   4 companies use this tag: AAPL, MSFT, TSLA, JNJ
#   Runner-up tag: GrossProfitPercentToSales
#   1 companies use this tag: JNJ

# InterestExpense:
#   Most common tag: InterestExpenseNonoperating
#   4 companies use this tag: TSLA, AMZN, JNJ, PG
#   Runner-up tag: InterestExpenseDebt
#   2 companies use this tag: INTU, WMT

# InventoryNet:
#   Most common tag: InventoryNet
#   7 companies use this tag: AAPL, MSFT, TSLA, WMT, AMZN, JNJ, PG
#   Runner-up tag: InventoryPartsAndComponentsNetOfReserves
#   1 companies use this tag: XOM

# LiabilitiesCurrent:
#   Most common tag: LiabilitiesCurrent
#   9 companies use this tag: INTU, AAPL, MSFT, TSLA, WMT, AMZN, JNJ, XOM, PG
#   Runner-up tag: AccruedLiabilitiesCurrent
#   4 companies use this tag: WMT, AMZN, JNJ, PG

# LongTermDebtNoncurrent:
#   Most common tag: LongTermDebtNoncurrent
#   7 companies use this tag: INTU, AAPL, MSFT, WMT, AMZN, JNJ, PG
#   Runner-up tag: LongTermDebtAndFinanceLeasesNoncurrent
#   1 companies use this tag: TSLA

# NetCashProvidedByUsedInOperatingActivities:
#   Most common tag: NetCashProvidedByUsedInOperatingActivities
#   10 companies use this tag: INTU, AAPL, MSFT, TSLA, JPM, WMT, AMZN, JNJ, XOM, PG
#   Runner-up tag: NetCashProvidedByUsedInInvestingActivities
#   10 companies use this tag: INTU, AAPL, MSFT, TSLA, JPM, WMT, AMZN, JNJ, XOM, PG

# NetIncomeLoss:
#   Most common tag: NetIncomeLoss
#   10 companies use this tag: INTU, AAPL, MSFT, TSLA, JPM, WMT, AMZN, JNJ, XOM, PG
#   Runner-up tag: OtherComprehensiveIncomeLossNetOfTax
#   7 companies use this tag: INTU, TSLA, WMT, AMZN, JNJ, XOM, PG

# OperatingIncomeLoss:
#   Most common tag: OperatingIncomeLoss
#   7 companies use this tag: INTU, AAPL, MSFT, TSLA, WMT, AMZN, PG
#   Runner-up tag: AdjustmentsNoncashItemsToReconcileNetIncomeLossToCashProvidedByUsedInOperatingActivities
#   1 companies use this tag: INTU

# PaymentsToAcquirePropertyPlantAndEquipment:
#   Most common tag: PaymentsToAcquirePropertyPlantAndEquipment
#   8 companies use this tag: INTU, AAPL, MSFT, TSLA, WMT, JNJ, XOM, PG

# Revenues:
#   Most common tag: Revenues
#   4 companies use this tag: INTU, WMT, XOM, PG
#   Runner-up tag: RevenuesNetOfInterestExpense
#   1 companies use this tag: JPM

# RevenueFromContractWithCustomerExcludingAssessedTax:
#   Most common tag: RevenueFromContractWithCustomerExcludingAssessedTax
#   6 companies use this tag: AAPL, MSFT, TSLA, WMT, AMZN, JNJ

# StockholdersEquity:
#   Most common tag: StockholdersEquity
#   9 companies use this tag: INTU, AAPL, MSFT, TSLA, JPM, WMT, AMZN, JNJ, XOM



# Total unique companies (CIK): 6,074
# AccountsReceivableNetCurrent: AccountsReceivableNetCurrent  Percentage of companies: 48.53%
# Assets: Assets  Percentage of companies: 99.61%
# AssetsCurrent: AssetsCurrent  Percentage of companies: 76.25%
# CashAndCashEquivalentsAtCarryingValue: CashAndCashEquivalentsAtCarryingValue  Percentage of companies: 77.26%
# CommonStockSharesOutstanding: CommonStockSharesOutstanding  Percentage of companies: 86.55%
# CostOfGoodsAndServicesSold: CostOfGoodsAndServicesSold  Percentage of companies: 30.54%
# DepreciationDepletionAndAmortization: DepreciationDepletionAndAmortization  Percentage of companies: 40.12%
# GrossProfit: GrossProfit  Percentage of companies: 35.09%
# InterestExpense: InterestExpenseNonoperating  Percentage of companies: 29.73%
# InventoryNet: InventoryNet  Percentage of companies: 38.21%
# LiabilitiesCurrent: LiabilitiesCurrent  Percentage of companies: 75.75%
# LongTermDebtNoncurrent: LongTermDebtNoncurrent  Percentage of companies: 24.88%
# NetCashProvidedByUsedInOperatingActivities: NetCashProvidedByUsedInOperatingActivities  Percentage of companies: 98.09%
# NetIncomeLoss: NetIncomeLoss  Percentage of companies: 93.15%
# OperatingIncomeLoss: OperatingIncomeLoss  Percentage of companies: 75.29%
# PaymentsToAcquirePropertyPlantAndEquipment: PaymentsToAcquirePropertyPlantAndEquipment  Percentage of companies: 57.33%
# Revenues: Revenues  Percentage of companies: 31.67%
# RevenueFromContractWithCustomerExcludingAssessedTax: RevenueFromContractWithCustomerExcludingAssessedTax  Percentage of companies: 43.52%
# StockholdersEquity: StockholdersEquity  Percentage of companies: 91.88%

