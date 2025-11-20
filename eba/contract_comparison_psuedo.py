"""
CONTRACT COMPARISON - FUNCTION 2
Compare proposed contract against master contract to identify new rules needed

OVERVIEW:
When a consumer proposes a new data contract, we need to determine:
1. Which rules are already covered by the existing master contract
2. Which rules are genuinely new and need to be added (THE DELTA)
3. Avoid adding duplicate rules even within the proposed contract

This function performs DELTA DETECTION ONLY. It does NOT create the merged contract.
Function 3 will handle merging and highlighting for human review.

INPUTS:
- Master Contract CSV (producer's golden copy - may or may not be consolidated)
- Proposed Contract CSV (consumer's new requirements)

OUTPUTS:
- New Rules CSV (ONLY rules that need to be added - the delta)
- Comparison Audit CSV (analysis of every PROPOSED rule - covered vs added)
- Summary JSON (statistics about comparison results)
"""

# =============================================================================
# DATA LOADING
# =============================================================================

def load_master_contract(csv_path):
    """
    Load the master (producer) contract CSV
    
    Parameters:
        csv_path: string - path to master contract CSV
        
    Returns:
        DataFrame with columns: Business Rule, Description, Business Term
    
    Business Logic:
        # Read CSV from file path
        # Return as pandas DataFrame
        # Expected columns: Business Rule, Description, Business Term
    """
    pass


def load_proposed_contract(csv_path):
    """
    Load the proposed (consumer) contract CSV
    
    Parameters:
        csv_path: string - path to proposed contract CSV
        
    Returns:
        DataFrame with columns: Business Rule, Description, Business Term
    
    Business Logic:
        # Read CSV from file path
        # Return as pandas DataFrame
        # Expected columns: Business Rule, Description, Business Term
    """
    pass


# =============================================================================
# DATA VALIDATION AND CLEANING
# =============================================================================

def validate_and_clean_dataframe(df, contract_type):
    """
    Apply validation and quality checks to the dataframe
    
    Parameters:
        df: DataFrame - raw input data
        contract_type: string - "master" or "proposed" (for logging)
        
    Returns:
        DataFrame - cleaned and validated
    
    Business Logic:
        # Check that required columns exist: Business Rule, Business Term
        # Remove rows where Business Rule is null or empty
        # Remove rows where Business Term is null or empty
        # Trim whitespace from Business Rule and Business Term columns
        # Remove exact duplicate rows (same Business Rule + same Business Term)
        # Log any validation issues or rows removed
        # Return cleaned dataframe
    """
    pass


# =============================================================================
# BUSINESS TERM EXTRACTION AND CATEGORIZATION
# =============================================================================

def categorize_business_terms(master_df, proposed_df):
    """
    Categorize business terms into existing and new
    
    Parameters:
        master_df: DataFrame - cleaned master contract
        proposed_df: DataFrame - cleaned proposed contract
        
    Returns:
        Dictionary with categorized terms:
        {
            "existing_terms": [list of terms in both contracts],
            "new_terms": [list of terms only in proposed contract]
        }
    
    Business Logic:
        # Get unique business terms from master contract
        # Get unique business terms from proposed contract
        
        # Find existing terms (intersection):
        # - Terms that appear in BOTH master and proposed
        
        # Find new terms:
        # - Terms that appear in proposed but NOT in master
        
        # Return dictionary with both lists
    """
    pass


def get_rules_for_business_term(df, business_term):
    """
    Filter dataframe to only rules for a specific business term
    
    Parameters:
        df: DataFrame - contract data
        business_term: string - the business term to filter for
        
    Returns:
        DataFrame - subset containing only rows for this business term
    
    Business Logic:
        # Filter dataframe where Business Term column equals the parameter
        # Return filtered dataframe
    """
    pass


# =============================================================================
# LLM COMPARISON - EXISTING BUSINESS TERMS
# =============================================================================

def compare_rules_for_existing_term(master_rules_df, proposed_rules_df, business_term, config):
    """
    Compare proposed rules against master rules for an existing business term
    Uses LLM to determine which proposed rules are new vs already covered
    
    Parameters:
        master_rules_df: DataFrame - all master rules for this business term
        proposed_rules_df: DataFrame - all proposed rules for this business term
        business_term: string - the business term being compared
        config: dict - configuration (model, region, etc.)
        
    Returns:
        Dictionary - structured JSON with comparison analysis
    
    Business Logic:
        # Extract list of master rules
        # Extract list of proposed rules
        
        # Build system prompt:
        #   - Based on Function 1 system prompt (for consistency)
        #   - Add comparison-specific instructions:
        #     - Identify duplicates WITHIN proposed rules first
        #     - For each unique proposed rule, compare to master rules:
        #       - Is it already covered? (semantically identical OR less strict)
        #       - Is it stricter? (flag as new_stricter)
        #       - Is it a different constraint type? (flag as new_different)
        #       - Is it unclear/conflicting? (flag as conflict)
        #   - Output format: JSON with analysis
        
        # Build user prompt:
        #   - Business term name
        #   - Master rules list
        #   - Proposed rules list
        #   - Explicit JSON structure expected
        
        # Attempt to call LLM (up to max_retries times):
        #   - Call Bedrock API with system prompt and user prompt
        #   - Model: from config
        #   - Temperature: 0 (for consistency)
        #   - Parse response as JSON
        #   - Validate JSON has required fields
        #   - If parsing fails, retry
        #   - If all retries fail, return error structure
        
        # Return JSON structure:
        # {
        #     "business_term": string,
        #     "master_rules_count": int,
        #     "proposed_rules_count": int,
        #     "new_rules_required": boolean,
        #     "analysis": [
        #         {
        #             "proposed_rule": string,
        #             "proposed_rule_index": int,
        #             "status": "covered" | "new_stricter" | "new_different" | "conflict" | "duplicate_within_proposed",
        #             "matched_master_rule": string or null,
        #             "matched_master_index": int or null,
        #             "canonical_rule": string (the rule to add if new, or the master rule if covered),
        #             "confidence": float (0.0 to 1.0),
        #             "reasoning": string
        #         }
        #     ],
        #     "notes": string
        # }
    """
    pass


# =============================================================================
# SELF-CONSOLIDATION - NEW BUSINESS TERMS
# =============================================================================

def consolidate_rules_for_new_term(proposed_rules_df, business_term, config):
    """
    Consolidate proposed rules for a NEW business term (not in master)
    Uses Function 1 logic to identify duplicates within proposed rules
    
    Parameters:
        proposed_rules_df: DataFrame - all proposed rules for this new term
        business_term: string - the business term
        config: dict - configuration
        
    Returns:
        Dictionary - structured JSON with consolidation analysis
    
    Business Logic:
        # Extract list of proposed rules
        
        # If only 1 rule:
        #   - Return simple structure marking it as unique/new
        #   - No LLM call needed
        
        # If 2+ rules:
        #   - Build system prompt (reuse Function 1 system prompt)
        #   - Build user prompt with rules list
        #   - Call LLM to identify duplicates
        #   - Parse response
        
        # Return JSON structure:
        # {
        #     "business_term": string,
        #     "proposed_rules_count": int,
        #     "unique_rules_identified": int,
        #     "duplicates_found": boolean,
        #     "is_new_business_term": true,
        #     "analysis": [
        #         {
        #             "proposed_rule": string,
        #             "proposed_rule_index": int,
        #             "status": "unique" | "duplicate",
        #             "canonical_rule": string,
        #             "confidence": float (0.0 to 1.0),
        #             "consolidated_with": [list of duplicate rules],
        #             "duplicate_of_index": int or null,
        #             "reasoning": string
        #         }
        #     ],
        #     "notes": string
        # }
    """
    pass


# =============================================================================
# ORCHESTRATION
# =============================================================================

def process_all_business_terms(master_df, proposed_df, categorized_terms, config):
    """
    Process all business terms from proposed contract
    
    Parameters:
        master_df: DataFrame - cleaned master contract
        proposed_df: DataFrame - cleaned proposed contract
        categorized_terms: dict - existing vs new terms
        config: dict - configuration
        
    Returns:
        List of dictionaries - comparison/consolidation results for all terms
    
    Business Logic:
        # Initialize empty results list
        
        # Process existing terms (terms in both contracts):
        # For each term in categorized_terms["existing_terms"]:
        #   - Get master rules for this term
        #   - Get proposed rules for this term
        #   - Call compare_rules_for_existing_term()
        #   - Append result to results list
        
        # Process new terms (terms only in proposed):
        # For each term in categorized_terms["new_terms"]:
        #   - Get proposed rules for this term
        #   - Call consolidate_rules_for_new_term()
        #   - Append result to results list
        
        # Return complete results list
    """
    pass


# =============================================================================
# OUTPUT GENERATION - NEW RULES (DELTA)
# =============================================================================

def create_new_rules_delta(all_results):
    """
    Create the new rules CSV containing ONLY rules that need to be added
    This is the delta between master and proposed contracts
    
    Parameters:
        all_results: list of dicts - comparison/consolidation results
        
    Returns:
        DataFrame - new rules that need to be added to master
    
    Business Logic:
        # Initialize list for new_rules
        
        # Process all comparison/consolidation results:
        # For each result in all_results:
        #   - Extract business_term
        #   - Determine if this is new business term or existing
        
        #   - For each rule analysis in result["analysis"]:
        #     - Check status
        #     
        #     - If status is "new_stricter", "new_different", or "conflict":
        #       - Add to new_rules with fields:
        #         - business_rule: canonical_rule from analysis
        #         - business_term
        #         - rule_type: status value (new_stricter, new_different, conflict)
        #         - related_master_rule: matched_master_rule from analysis (can be null)
        #         - confidence: from analysis
        #         - reasoning: from analysis
        #     
        #     - If status is "unique" (from new business term consolidation):
        #       - Add to new_rules with fields:
        #         - business_rule: canonical_rule from analysis
        #         - business_term
        #         - rule_type: "new_business_term"
        #         - related_master_rule: null (no master rule for new terms)
        #         - confidence: from analysis
        #         - reasoning: from analysis
        #     
        #     - If status is "covered" or "duplicate_within_proposed":
        #       - Do NOT add to new_rules (not part of delta)
        
        # Convert new_rules list to DataFrame
        # Sort by business_term, then by rule_type, then by business_rule
        # Return DataFrame
        #
        # NOTE: If no new rules needed, return empty DataFrame (headers only)
    """
    pass


# =============================================================================
# OUTPUT GENERATION - COMPARISON AUDIT
# =============================================================================

def create_comparison_audit(all_results):
    """
    Create detailed audit trail of every PROPOSED rule
    Shows disposition: covered, duplicate_within_proposed, or added as new
    Does NOT include master contract rules
    
    Parameters:
        all_results: list of dicts - comparison/consolidation results
        
    Returns:
        DataFrame - comparison audit trail for all proposed rules
    
    Business Logic:
        # Initialize list for audit_records
        
        # Process all results:
        # For each result in all_results:
        #   - Extract business_term
        #   - Determine if new business term or existing
        
        #   - For each rule analysis in result["analysis"]:
        #     - Create audit record with fields:
        #       - proposed_rule: exact text from proposed contract
        #       - business_term
        #       - status: from analysis
        #         Possible values:
        #         - "covered" (already in master)
        #         - "duplicate_within_proposed" (duplicate of another proposed rule)
        #         - "new_stricter" (added - more restrictive than master)
        #         - "new_different" (added - different constraint type)
        #         - "conflict" (added - conflicting with master)
        #         - "unique" (added - from new business term, no duplicates)
        #         - "duplicate" (not added - duplicate within new business term)
        #       - related_master_rule: matched_master_rule from analysis (null if no match)
        #       - canonical_rule: the rule text (canonical form)
        #       - confidence: from analysis
        #       - is_new_business_term: boolean (true if term not in master)
        #       - added_to_new_rules: boolean (true if status indicates rule was added)
        #       - reasoning: from analysis
        #     - Append to audit_records list
        
        # Convert audit_records list to DataFrame
        # Sort by added_to_new_rules (desc), then business_term, then proposed_rule
        #   (This groups new rules at top for easy review)
        # Return DataFrame
    """
    pass


# =============================================================================
# OUTPUT GENERATION - SUMMARY STATISTICS
# =============================================================================

def generate_summary_statistics(new_rules_df, audit_df, master_original_count, proposed_original_count):
    """
    Generate summary statistics about the comparison
    
    Parameters:
        new_rules_df: DataFrame - new rules that need to be added (delta)
        audit_df: DataFrame - comparison audit of all proposed rules
        master_original_count: int - count of rules in original master contract
        proposed_original_count: int - count of rules in original proposed contract
        
    Returns:
        Dictionary - summary statistics
    
    Business Logic:
        # Calculate statistics:
        # - new_rules_required: boolean (len(new_rules_df) > 0)
        # - total_proposed_rules: proposed_original_count
        # - total_master_rules: master_original_count
        # - already_covered: count of audit_df where status="covered"
        # - duplicates_within_proposed: count of audit_df where status="duplicate_within_proposed" OR status="duplicate"
        # - new_rules_added: len(new_rules_df)
        # - breakdown:
        #   - new_stricter: count where rule_type="new_stricter"
        #   - new_different: count where rule_type="new_different"
        #   - new_business_term: count where rule_type="new_business_term"
        #   - conflict: count where rule_type="conflict"
        
        # Return dictionary with all statistics
    """
    pass


# =============================================================================
# FILE OUTPUT
# =============================================================================

def save_outputs(new_rules_df, audit_df, summary, output_dir):
    """
    Save all output files
    
    Parameters:
        new_rules_df: DataFrame - new rules to be added (delta)
        audit_df: DataFrame - comparison audit of proposed rules
        summary: dict - summary statistics
        output_dir: string - output directory path
        
    Returns:
        None
    
    Business Logic:
        # Create output directory if doesn't exist
        
        # Save new rules (delta):
        # - Save to CSV: output_dir/new_rules.csv
        # - Print confirmation
        # - If empty, print "No new rules required"
        
        # Save comparison audit:
        # - Save to CSV: output_dir/comparison_audit.csv
        # - Print confirmation
        
        # Try to save Excel with formatting:
        # - If openpyxl available:
        #   - Save audit to Excel: output_dir/comparison_audit_formatted.xlsx
        #   - Apply formatting:
        #     - Color coding based on status:
        #       - Green: covered
        #       - Yellow: new_stricter, new_different, conflict
        #       - Blue: new_business_term
        #       - Gray: duplicate_within_proposed
        #   - Bold rules where added_to_new_rules=TRUE
        #   - Adjust column widths
        #   - Save workbook
        # - If openpyxl not available, print message
        
        # Save summary statistics:
        # - Save to JSON: output_dir/comparison_summary.json
        # - Print confirmation
        
        # Print summary to console:
        # - New rules required? Yes/No
        # - If yes:
        #   - Number of new rules to add
        #   - Breakdown by type
        #   - Path to new_rules.csv
        # - Number of proposed rules analyzed
        # - Number already covered
        # - Number of duplicates within proposed
    """
    pass


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    """
    Execute the complete contract comparison workflow
    
    Business Logic:
        # STEP 1: LOAD DATA
        # - Call load_master_contract() with master CSV path
        # - Store as master_df
        # - Call load_proposed_contract() with proposed CSV path
        # - Store as proposed_df
        # - Store original counts for statistics
        
        # STEP 2: VALIDATE AND CLEAN
        # - Call validate_and_clean_dataframe(master_df, "master")
        # - Store as master_df_clean
        # - Call validate_and_clean_dataframe(proposed_df, "proposed")
        # - Store as proposed_df_clean
        
        # STEP 3: CATEGORIZE BUSINESS TERMS
        # - Call categorize_business_terms(master_df_clean, proposed_df_clean)
        # - Store as categorized_terms
        # - Print summary of categorization
        
        # STEP 4: PROCESS ALL BUSINESS TERMS
        # - Call process_all_business_terms(master_df_clean, proposed_df_clean, categorized_terms, config)
        # - Store as all_results
        
        # STEP 5: CREATE OUTPUTS
        # - Call create_new_rules_delta(all_results)
        # - Store as new_rules_df
        # - Call create_comparison_audit(all_results)
        # - Store as audit_df
        # - Call generate_summary_statistics(new_rules_df, audit_df, master_count, proposed_count)
        # - Store as summary
        
        # STEP 6: SAVE OUTPUTS
        # - Call save_outputs(new_rules_df, audit_df, summary, output_dir)
        
        # STEP 7: PRINT FINAL SUMMARY
        # - Print completion message
        # - Print key statistics
        # - If new rules required:
        #   - Print "New rules file: output/new_rules.csv"
        #   - Print "Ready for Function 3 (merge and highlight)"
        # - If no new rules:
        #   - Print "No new rules required - proposed contract fully covered by master"
    """
    pass


# =============================================================================
# EXPECTED LLM JSON OUTPUT - COMPARISON FOR EXISTING TERM
# =============================================================================

"""
EXAMPLE 1: Comparing rules for existing business term (CUSIP Identifier)
--------------------------------------------------------------------------

Master Contract Rules:
- "CUSIP must be exactly 9 characters in length"
- "CUSIP must contain only alphanumeric characters"

Proposed Contract Rules:
- "CUSIP must be 9 characters"
- "The CUSIP code must be 9 chars"
- "CUSIP cannot be null"
- "CUSIP must be exactly 9 alphanumeric characters"

Expected LLM JSON Output:
{
    "business_term": "CUSIP Identifier",
    "master_rules_count": 2,
    "proposed_rules_count": 4,
    "new_rules_required": true,
    "analysis": [
        {
            "proposed_rule": "CUSIP must be 9 characters",
            "proposed_rule_index": 0,
            "status": "covered",
            "matched_master_rule": "CUSIP must be exactly 9 characters in length",
            "matched_master_index": 0,
            "canonical_rule": "CUSIP must be exactly 9 characters in length",
            "confidence": 1.0,
            "reasoning": "This proposed rule is semantically identical to the master rule at index 0. Both enforce the 9-character length requirement."
        },
        {
            "proposed_rule": "The CUSIP code must be 9 chars",
            "proposed_rule_index": 1,
            "status": "duplicate_within_proposed",
            "matched_master_rule": "CUSIP must be exactly 9 characters in length",
            "matched_master_index": 0,
            "canonical_rule": "CUSIP must be exactly 9 characters in length",
            "confidence": 1.0,
            "reasoning": "This is a duplicate of proposed rule at index 0 and is also covered by master rule at index 0."
        },
        {
            "proposed_rule": "CUSIP cannot be null",
            "proposed_rule_index": 2,
            "status": "new_different",
            "matched_master_rule": null,
            "matched_master_index": null,
            "canonical_rule": "CUSIP cannot be null",
            "confidence": 1.0,
            "reasoning": "This is a different constraint type (nullability) not present in the master contract. Master only has length and character type constraints."
        },
        {
            "proposed_rule": "CUSIP must be exactly 9 alphanumeric characters",
            "proposed_rule_index": 3,
            "status": "covered",
            "matched_master_rule": "CUSIP must be exactly 9 characters in length",
            "matched_master_index": 0,
            "canonical_rule": "CUSIP must be exactly 9 characters in length",
            "confidence": 1.0,
            "reasoning": "This combines both master rules (9 characters + alphanumeric). Both constraints are already covered by the master contract."
        }
    ],
    "notes": "One new rule required for nullability constraint. Two proposed rules are duplicates of each other."
}


EXAMPLE 2: Stricter rule scenario
-----------------------------------

Master Contract Rules:
- "Age must be greater than 18"

Proposed Contract Rules:
- "Age must be greater than 21"
- "Age must be at least 18"

Expected LLM JSON Output:
{
    "business_term": "Customer Age",
    "master_rules_count": 1,
    "proposed_rules_count": 2,
    "new_rules_required": true,
    "analysis": [
        {
            "proposed_rule": "Age must be greater than 21",
            "proposed_rule_index": 0,
            "status": "new_stricter",
            "matched_master_rule": "Age must be greater than 18",
            "matched_master_index": 0,
            "canonical_rule": "Age must be greater than 21",
            "confidence": 1.0,
            "reasoning": "This rule is more restrictive than the master rule. Master allows age > 18, but this requires age > 21. Flag for review as a stricter requirement."
        },
        {
            "proposed_rule": "Age must be at least 18",
            "proposed_rule_index": 1,
            "status": "covered",
            "matched_master_rule": "Age must be greater than 18",
            "matched_master_index": 0,
            "canonical_rule": "Age must be greater than 18",
            "confidence": 0.95,
            "reasoning": "This is semantically covered by the master rule. Master rule 'greater than 18' is stricter than 'at least 18' (>= vs >), so master rule satisfies this requirement."
        }
    ],
    "notes": "One stricter rule requires review."
}


EXAMPLE 3: Conflict scenario
------------------------------

Master Contract Rules:
- "Balance must be between 0 and 100"

Proposed Contract Rules:
- "Balance must be greater than 10"

Expected LLM JSON Output:
{
    "business_term": "Account Balance",
    "master_rules_count": 1,
    "proposed_rules_count": 1,
    "new_rules_required": true,
    "analysis": [
        {
            "proposed_rule": "Balance must be greater than 10",
            "proposed_rule_index": 0,
            "status": "conflict",
            "matched_master_rule": "Balance must be between 0 and 100",
            "matched_master_index": 0,
            "canonical_rule": "Balance must be greater than 10",
            "confidence": 0.85,
            "reasoning": "This rule has partial overlap with master. It's stricter on the lower bound (10 vs 0) but removes the upper bound (no max vs 100 max). This creates a conflict that needs human review."
        }
    ],
    "notes": "Conflict detected: proposed rule is stricter in one dimension but more permissive in another."
}
"""


# =============================================================================
# EXPECTED LLM JSON OUTPUT - CONSOLIDATION FOR NEW TERM
# =============================================================================

"""
EXAMPLE: New business term with multiple rules
------------------------------------------------

Proposed Contract Rules for "ESG Score" (NEW term not in master):
- "ESG Score must be between 0 and 100"
- "ESG score should be from 0 to 100"
- "ESG Score cannot be null"

Expected LLM JSON Output (using Function 1 logic):
{
    "business_term": "ESG Score",
    "proposed_rules_count": 3,
    "unique_rules_identified": 2,
    "duplicates_found": true,
    "is_new_business_term": true,
    "analysis": [
        {
            "proposed_rule": "ESG Score must be between 0 and 100",
            "proposed_rule_index": 0,
            "status": "unique",
            "canonical_rule": "ESG Score must be between 0 and 100",
            "confidence": 1.0,
            "consolidated_with": [],
            "duplicate_of_index": null,
            "reasoning": "Selected as canonical for the range constraint. Clear and uses 'must' for strong requirement."
        },
        {
            "proposed_rule": "ESG score should be from 0 to 100",
            "proposed_rule_index": 1,
            "status": "duplicate",
            "canonical_rule": "ESG Score must be between 0 and 100",
            "confidence": 0.95,
            "consolidated_with": ["ESG Score must be between 0 and 100"],
            "duplicate_of_index": 0,
            "reasoning": "Semantically identical to rule at index 0. Same range constraint (0-100), weaker modal 'should' vs 'must'."
        },
        {
            "proposed_rule": "ESG Score cannot be null",
            "proposed_rule_index": 2,
            "status": "unique",
            "canonical_rule": "ESG Score cannot be null",
            "confidence": 1.0,
            "consolidated_with": [],
            "duplicate_of_index": null,
            "reasoning": "Different constraint type (nullability). Unique requirement separate from range validation."
        }
    ],
    "notes": "New business term. Two unique rules required after deduplication."
}
"""
