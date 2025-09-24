#!/usr/bin/env python3
"""
Linguistic Analysis for Personality Dataset
Analyzes text samples and generates scores for 6 psychological categories on a 1-5 scale.
This version ensures all text content is properly preserved in the output.
"""

import re
import os
from collections import defaultdict, Counter

def analyze_linguistic_features(text):
    """
    Analyze linguistic features from text and return feature metrics.

    Args:
        text (str): Input text to analyze

    Returns:
        dict: Dictionary containing linguistic feature rates and metrics
    """
    if not text or len(text.strip()) == 0:
        return None

    # Clean text - remove URLs and multiple separators for analysis (but keep original)
    clean_text = re.sub(r'\|\|\|', ' ', text)
    clean_text = re.sub(r'https?://[^\s]+', '', clean_text).strip()

    # Split into sentences and words
    sentences = [s.strip() for s in re.split(r'[.!?]+', clean_text) if len(s.strip()) > 3]
    words = [w.lower() for w in re.findall(r'\b\w+\b', clean_text) if len(w) > 0]

    if len(words) == 0 or len(sentences) == 0:
        return None

    # Define word lists for analysis
    first_person_pronouns = ['i', 'me', 'my', 'mine', 'myself']
    second_person_pronouns = ['you', 'your', 'yours', 'yourself']
    positive_words = ['good', 'great', 'excellent', 'perfect', 'best', 'love', 'like',
                      'enjoy', 'happy', 'awesome', 'amazing', 'wonderful', 'efficient',
                      'reliable', 'nice', 'beautiful', 'fantastic', 'brilliant',
                      'outstanding', 'superb', 'marvelous', 'terrific', 'fabulous']
    social_words = ['we', 'us', 'our', 'together', 'friend', 'team', 'group',
                    'community', 'people', 'guys', 'everyone', 'others', 'folks',
                    'colleagues', 'society', 'family', 'friends']

    # Count feature occurrences
    first_person_count = sum(1 for word in words if word in first_person_pronouns)
    second_person_count = sum(1 for word in words if word in second_person_pronouns)
    positive_word_count = sum(1 for word in words if word in positive_words)
    social_word_count = sum(1 for word in words if word in social_words)

    # Count punctuation marks
    exclamation_count = len(re.findall(r'!', text))
    question_count = len(re.findall(r'\?', text))

    # Calculate basic metrics
    total_words = len(words)
    total_sentences = len(sentences)
    unique_words = len(set(words))

    return {
        'first_person_rate': (first_person_count / total_words) * 100,
        'second_person_rate': (second_person_count / total_words) * 100,
        'exclamation_rate': (exclamation_count / total_sentences) * 100,
        'question_rate': (question_count / total_sentences) * 100,
        'positive_word_rate': (positive_word_count / total_words) * 100,
        'social_word_rate': (social_word_count / total_words) * 100,
        'avg_sentence_length': total_words / total_sentences,
        'lexical_diversity': (unique_words / total_words) * 100,
        'total_words': total_words,
        'total_sentences': total_sentences
    }

def calculate_scores_1_to_5(features):
    """
    Calculate psychological scores for 6 categories on a 1-5 scale.

    Args:
        features (dict): Dictionary of linguistic features

    Returns:
        dict: Dictionary containing scores for each psychological category
    """
    if not features:
        return None

    # Calculate raw scores and scale to 1-5 range
    scores = {
        'social_interaction': min(5, max(1,
                                         (features['second_person_rate'] * 1.0) +
                                         (features['question_rate'] * 0.25) +
                                         (features['social_word_rate'] * 0.75) + 0.5
                                         )),

        'communication': min(5, max(1,
                                    (features['lexical_diversity'] * 0.075) +
                                    (min(features['avg_sentence_length'], 20) * 0.1) + 1
                                    )),

        'attention': min(5, max(1,
                                (features['question_rate'] * 0.3) +
                                (features['second_person_rate'] * 0.75) + 0.5
                                )),

        'group_comfort': min(5, max(1,
                                    (features['social_word_rate'] * 1.0) +
                                    (features['positive_word_rate'] * 0.75) +
                                    (features['second_person_rate'] * 0.4) + 0.5
                                    )),

        'energy_source': min(5, max(1,
                                    (features['exclamation_rate'] * 0.15) +
                                    (features['positive_word_rate'] * 0.6) +
                                    (features['first_person_rate'] * 0.4) + 1
                                    )),

        'self_expression': min(5, max(1,
                                      (features['first_person_rate'] * 0.75) +
                                      (features['lexical_diversity'] * 0.05) + 0.5
                                      ))
    }

    # Round scores to 1 decimal place
    for key in scores:
        scores[key] = round(scores[key], 1)

    return scores

def try_read_file(file_path):
    """
    Try to read file with different encodings.

    Args:
        file_path (str): Path to the file

    Returns:
        str: File content or None if failed
    """
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
                print(f"Successfully read file with {encoding} encoding")
                return content
        except UnicodeDecodeError:
            continue
        except Exception as e:
            continue

    print("Could not read file with any encoding")
    return None

def detect_file_format(content):
    """
    Detect the file format and separator.

    Args:
        content (str): File content

    Returns:
        str: Detected separator
    """
    lines = content.split('\n')[:10]  # Check first 10 lines

    tab_count = sum(line.count('\t') for line in lines)
    comma_count = sum(line.count(',') for line in lines)

    if tab_count > comma_count:
        return '\t'
    else:
        return ','

def clean_text_for_csv(text):
    """
    Clean text for CSV output while preserving readability.

    Args:
        text (str): Original text

    Returns:
        str: Cleaned text suitable for CSV
    """
    # Replace problematic characters but keep the text readable
    cleaned = text.replace('\t', ' ')  # Replace tabs with spaces
    cleaned = cleaned.replace('\n', ' ')  # Replace newlines with spaces
    cleaned = cleaned.replace('\r', ' ')  # Replace carriage returns with spaces
    cleaned = cleaned.replace('|||', ' | ')  # Make separators readable
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Collapse multiple spaces
    cleaned = cleaned.strip()  # Remove leading/trailing whitespace

    # Remove or replace problematic characters that might break CSV
    cleaned = cleaned.replace('"', "'")  # Replace quotes

    return cleaned

def load_dataset(file_path):
    """
    Load and parse the dataset from various file formats.

    Args:
        file_path (str): Path to the dataset file

    Returns:
        list: List of dictionaries containing parsed data
    """
    try:
        print(f"Attempting to read file: {file_path}")

        # Try to read with different encodings
        content = try_read_file(file_path)
        if not content:
            return []

        print(f"File size: {len(content)} characters")

        # Detect format
        separator = detect_file_format(content)
        tab_char = '\t'
        print(f"Using separator: {'TAB' if separator == tab_char else 'COMMA'}")

        # Split into lines
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        print(f"Found {len(lines)} non-empty lines")

        processed_data = []

        for i, line in enumerate(lines):
            # Split by detected separator
            if separator == '\t':
                parts = line.split('\t')
            else:
                parts = line.split(',')

            # Show first few lines for debugging
            if i < 3:
                print(f"Line {i+1}: {len(parts)} parts")
                print(f"  First part preview: '{parts[0][:100]}...'")

            if len(parts) >= 1:  # At least one part
                # Combine all parts to get the full text
                full_text = separator.join(parts) if len(parts) > 1 else parts[0]

                # Look for 4-letter personality type at start
                type_match = re.search(r'\b([A-Z]{4})\b', full_text)
                if type_match:
                    personality_type = type_match.group(1)

                    # Extract text - everything after the personality type
                    text_start = full_text.find(personality_type) + 4
                    text = full_text[text_start:].strip()

                    # Only add if there's actual text content
                    if text and len(text) > 10:  # Ensure meaningful content
                        processed_data.append({
                            'type': personality_type,
                            'text': text
                        })

                        if len(processed_data) <= 3:  # Show first few entries
                            print(f"Processed entry {len(processed_data)}:")
                            print(f"  Type: {personality_type}")
                            print(f"  Text length: {len(text)} characters")
                            print(f"  Text preview: '{text[:150]}...'")
                            print()

        print(f"Successfully processed {len(processed_data)} entries")
        return processed_data

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return []

def process_dataset(file_path):
    """
    Process the entire dataset and return results with scores.

    Args:
        file_path (str): Path to the dataset file

    Returns:
        list: List of dictionaries containing analysis results
    """
    # Load the dataset
    data = load_dataset(file_path)

    if not data:
        print("No data loaded. Please check your file.")
        return []

    print(f"\nAnalyzing {len(data)} entries...")
    results = []

    for i, entry in enumerate(data):
        # Analyze features and calculate scores
        features = analyze_linguistic_features(entry['text'])
        scores = calculate_scores_1_to_5(features)

        if scores:
            results.append({
                'index': i + 1,
                'type': entry['type'],
                'text': entry['text'],  # Keep original text
                'social_interaction': scores['social_interaction'],
                'communication': scores['communication'],
                'attention': scores['attention'],
                'group_comfort': scores['group_comfort'],
                'energy_source': scores['energy_source'],
                'self_expression': scores['self_expression']
            })
        else:
            print(f"Warning: Could not analyze entry {i+1} (Type: {entry['type']})")

    print(f"Successfully analyzed {len(results)} entries")
    return results

def create_new_dataset(results, output_file):
    """
    Create new dataset with scores appended after user IDs instead of text.

    Args:
        results (list): List of analysis results
        output_file (str): Output file path
    """
    try:
        print(f"\nCreating output file: {output_file}")

        with open(output_file, 'w', encoding='utf-8') as file:
            # Write header
            header = "type\tuser_id\tsocial_interaction\tcommunication\tattention\tgroup_comfort\tenergy_source\tself_expression\n"
            file.write(header)

            # Write data
            for i, result in enumerate(results):
                # Use simple user ID instead of text
                user_id = f"user {i + 1}"

                # Create the row
                row = f"{result['type']}\t{user_id}\t{result['social_interaction']}\t{result['communication']}\t{result['attention']}\t{result['group_comfort']}\t{result['energy_source']}\t{result['self_expression']}\n"

                file.write(row)

                # Show progress for first few entries
                if i < 3:
                    print(f"Sample output row {i+1}:")
                    print(f"  Type: {result['type']}")
                    print(f"  User ID: {user_id}")
                    print(f"  Scores: SI={result['social_interaction']}, COM={result['communication']}, ATT={result['attention']}")
                    print()

        print(f"Successfully saved {len(results)} entries to {output_file}")

        # Verify the file was created properly
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"Output file size: {file_size} bytes")

            # Show a preview of the created file
            print(f"\nFirst few lines of {output_file}:")
            with open(output_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i < 5:  # Show first 5 lines
                        print(f"{i+1}: {line.strip()}")
                    else:
                        break

    except Exception as e:
        print(f"Error saving dataset: {e}")
        import traceback
        traceback.print_exc()

def display_results(results):
    """
    Display analysis results in a formatted table.

    Args:
        results (list): List of analysis results
    """
    print(f"\n{'='*60}")
    print("LINGUISTIC ANALYSIS RESULTS (1-5 Scale)")
    print(f"{'='*60}")
    print(f"Total entries processed: {len(results)}")

    # Show first 20 results
    print(f"\nFirst {min(20, len(results))} Results:")
    print("Index | Type | User ID | SI  | COM | ATT | GC  | ES  | SE")
    print("------|------|---------|-----|-----|-----|-----|-----|-----")

    for result in results[:20]:
        user_id = f"user {result['index']}"
        print(f"{result['index']:5} | {result['type']} | {user_id:7} | {result['social_interaction']:3.1f} | "
              f"{result['communication']:4.1f} | {result['attention']:4.1f} | {result['group_comfort']:3.1f} | "
              f"{result['energy_source']:3.1f} | {result['self_expression']:3.1f}")

    if len(results) > 20:
        print(f"... and {len(results) - 20} more entries")

def calculate_basic_stats(values):
    """Calculate basic statistics for a list of values."""
    if not values:
        return {}

    values = sorted(values)
    n = len(values)

    return {
        'count': n,
        'mean': sum(values) / n,
        'min': min(values),
        'max': max(values),
        'median': values[n//2] if n % 2 == 1 else (values[n//2-1] + values[n//2]) / 2
    }

def display_statistics(results):
    """
    Display statistical analysis of the results.

    Args:
        results (list): List of analysis results
    """
    try:
        # Group by personality type
        type_groups = defaultdict(list)
        for result in results:
            type_groups[result['type']].append(result)

        # Calculate averages by personality type
        print(f"\n{'='*70}")
        print("AVERAGES BY PERSONALITY TYPE")
        print(f"{'='*70}")
        print("Type | Count | SI  | COM | ATT | GC  | ES  | SE")
        print("-----|-------|-----|-----|-----|-----|-----|-----")

        for ptype in sorted(type_groups.keys()):
            group = type_groups[ptype]
            count = len(group)
            si_avg = sum(r['social_interaction'] for r in group) / count
            com_avg = sum(r['communication'] for r in group) / count
            att_avg = sum(r['attention'] for r in group) / count
            gc_avg = sum(r['group_comfort'] for r in group) / count
            es_avg = sum(r['energy_source'] for r in group) / count
            se_avg = sum(r['self_expression'] for r in group) / count

            print(f"{ptype} | {count:5} | {si_avg:3.1f} | {com_avg:4.1f} | {att_avg:4.1f} | {gc_avg:3.1f} | {es_avg:3.1f} | {se_avg:3.1f}")

        # Overall statistics
        print(f"\n{'='*70}")
        print("OVERALL STATISTICS")
        print(f"{'='*70}")

        categories = ['social_interaction', 'communication', 'attention', 'group_comfort', 'energy_source', 'self_expression']

        for category in categories:
            values = [result[category] for result in results]
            stats = calculate_basic_stats(values)
            print(f"{category.replace('_', ' ').title()}: Mean={stats['mean']:.2f}, Min={stats['min']:.1f}, Max={stats['max']:.1f}, Median={stats['median']:.2f}")

        # Type distribution
        print(f"\n{'='*70}")
        print("PERSONALITY TYPE DISTRIBUTION")
        print(f"{'='*70}")

        type_counts = Counter(result['type'] for result in results)
        for ptype in sorted(type_counts.keys()):
            count = type_counts[ptype]
            percentage = (count / len(results)) * 100
            print(f"{ptype}: {count:3} entries ({percentage:5.1f}%)")

    except Exception as e:
        print(f"Error displaying statistics: {e}")

def main():
    """Main function to run the linguistic analysis."""
    print("Linguistic Analysis for Personality Dataset")
    print("=" * 50)

    # Try different possible file names
    possible_files = [
        '../dataset/tests_dataset.csv'
    ]

    input_file = None
    for filename in possible_files:
        if os.path.exists(filename):
            input_file = filename
            break

    if not input_file:
        print("Could not find dataset file. Looking for:")
        for filename in possible_files:
            print(f"  - {filename}")
        print("\nPlease make sure your dataset file is in the current directory.")
        return

    output_file = 'tests_dataset_with_scores.csv'

    # Process the dataset
    print(f"Loading dataset from: {input_file}")
    results = process_dataset(input_file)

    if not results:
        print("No valid data found. Please check your input file format.")
        return

    # Display results
    display_results(results)

    # Display statistics
    display_statistics(results)

    # Create new dataset with scores
    create_new_dataset(results, output_file)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"Original entries: {len(results)}")
    print(f"Output file: {output_file}")
    print("\nCategory Legend:")
    print("- SI: Social Interaction (engagement with others)")
    print("- COM: Communication (language sophistication)")
    print("- ATT: Attention (focus on others)")
    print("- GC: Group Comfort (collaborative orientation)")
    print("- ES: Energy Source (expressiveness and assertion)")
    print("- SE: Self Expression (personal disclosure)")

    print(f"\nYou can now open '{output_file}' to see the complete dataset with user IDs and scores!")
    print("The output format is: type | user_id | social_interaction | communication | attention | group_comfort | energy_source | self_expression")

if __name__ == "__main__":
    main()