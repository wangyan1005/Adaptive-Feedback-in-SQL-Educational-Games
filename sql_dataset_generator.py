import openai
import csv
import json
import random
import time
import os
from typing import List, Dict

class SQLDatasetGenerator:
    """SQL Error Dataset Generator with Balanced Distribution"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key="replace_with_your_api_key")
        self.model = model
        
        # Generation statistics
        self.generation_stats = {
            'total_attempts': 0,
            'successful_generations': 0,
            'api_failures': 0,
            'fallback_used': 0
        }
        
        # Database schema
        self.schema = {
            'Employees': ['Employee_ID', 'Name', 'Job_Role', 'Division', 'Last_Login_Time'],
            'Robots': ['Robot_ID', 'Model', 'Manufacturing_Date', 'Status', 'Last_Software_Update', 'Employee_ID'],
            'Logs': ['Log_ID', 'Employee_ID', 'Action_Description', 'Timestamp', 'Robot_ID'],
            'Incidents': ['Incident_ID', 'Description', 'Timestamp', 'Robot_ID', 'Employee_ID'],
            'Access_Codes': ['Access_Code_ID', 'Employee_ID', 'Level_of_Access', 'Timestamp_of_Last_Use']
        }
        
        # Complete error taxonomy
        self.error_taxonomy = {
            'syntax': [
                'misspelling',
                'missing quotes', 
                'missing commas',
                'missing semicolons',
                'non-standard operators',
                'unmatched brackets',
                'data type mismatch'
            ],
            'schema': [
                'undefined column',
                'undefined table', 
                'undefined function',
                'ambiguous column in multi-table queries'
            ],
            'logic': [
                'aggregation misuse',
                'incorrect GROUP BY usage',
                'incorrect HAVING clause',
                'incorrect JOIN usage',
                'incorrect ORDER BY usage', 
                'incorrect DISTINCT usage',
                'incorrect SELECT usage',
                'operator Error'
            ],
            'others': [
                'incorrect wildcard usage',
                'inefficient query patterns'
            ]
        }

    def calculate_balanced_distribution(self, total_records: int) -> Dict[str, Dict[str, int]]:
        """Calculate balanced distribution across all subtypes"""
        distribution = {}
        
        # Calculate total number of subtypes
        total_subtypes = sum(len(subtypes) for subtypes in self.error_taxonomy.values())
        
        # Base allocation: minimum records per subtype
        base_allocation = total_records // total_subtypes
        remaining_records = total_records % total_subtypes
        
        print(f" Balance Distribution Plan:")
        print(f" Total subtypes: {total_subtypes}")
        print(f" Base allocation per subtype: {base_allocation}")
        print(f" Remaining records to distribute: {remaining_records}")
        
        # Allocate records for each error_type and subtype
        subtype_list = []
        for error_type, subtypes in self.error_taxonomy.items():
            distribution[error_type] = {}
            for subtype in subtypes:
                subtype_list.append((error_type, subtype))
        
        # Randomly distribute remaining records
        random.shuffle(subtype_list)
        
        for i, (error_type, subtype) in enumerate(subtype_list):
            allocation = base_allocation
            if i < remaining_records:
                allocation += 1  # Give extra record to first N subtypes
            distribution[error_type][subtype] = allocation
        
        # Print allocation plan
        print(f"\n Detailed Distribution:")
        for error_type, subtypes in distribution.items():
            total_for_type = sum(subtypes.values())
            print(f"   {error_type}: {total_for_type} records")
            for subtype, count in subtypes.items():
                print(f"     - {subtype}: {count}")
        
        return distribution

    def create_detailed_prompt(self, error_type: str, error_subtype: str) -> str:
        """Create detailed prompt for API generation"""
        schema_text = "\n".join([f"{table}: {', '.join(columns)}" 
                                for table, columns in self.schema.items()])
        
        # Specific guidance for each error subtype
        error_guidance = {
            'syntax': {
                'misspelling': "Misspell a SQL keyword: SELECT→SELCT, FROM→FORM, WHERE→WERE, GROUP BY→GROPU BY, ORDER BY→ODERR BY",
                'missing quotes': "Forget quotes around string values in WHERE clause (e.g., WHERE Name = John instead of 'John')",
                'missing commas': "Omit commas between column names in SELECT (e.g., SELECT Name Job_Role FROM...)",
                'missing semicolons': "Don't end the SQL statement with semicolon",
                'non-standard operators': "Use incorrect operators like !== ",
                'unmatched brackets': "Have opening parenthesis without closing one (e.g., WHERE (Employee_ID = 1)",
                'data type mismatch': "Use string value where integer expected (e.g., WHERE Employee_ID = 'text')"
            },
            'schema': {
                'undefined column': "Reference a column that doesn't exist (use names like 'Salary', 'Time', 'Department')",
                'undefined table': "Query a table that doesn't exist (use names like 'Worker', 'Robot')",
                'undefined function': "Use a function that doesn't exist (like AGGREGATE(), LOW())",
                'ambiguous column in multi-table queries': "Use Employee_ID without table qualification when multiple tables have this column"
            },
            'logic': {
                'aggregation misuse': "Use aggregate function with non-grouped columns (e.g., SELECT Name, COUNT(*) FROM Employees)",
                'incorrect GROUP BY usage': "GROUP BY wrong column or missing aggregate in SELECT with GROUP BY",
                'incorrect HAVING clause': "Use HAVING without GROUP BY or use it like WHERE clause",
                'incorrect JOIN usage': "Use JOIN without ON clause or with wrong join condition",
                'incorrect ORDER BY usage': "ORDER BY non-selected column in aggregate query or use invalid column",
                'incorrect DISTINCT usage': "Use DISTINCT incorrectly with aggregate functions or in wrong position",
                'incorrect SELECT usage': "SELECT with mixed aggregate and non-aggregate without GROUP BY",
                'operator Error': "Like Mix AND OR incorrectly"
            },
            'others': {
                'incorrect wildcard usage': "Use wildcards incorrectly (%, _ in wrong context or wrong operators)",
                'inefficient query patterns': "unfinshed query like SELECT * FROM Employees WHERE"
            }
        }
        
        specific_guidance = error_guidance.get(error_type, {}).get(error_subtype, "Create appropriate error for this category")
        target_emotion = random.choice(['Anger', 'Happiness', 'Calmness', 'Sadness', 'Neutral'])
        
        # Emotion-specific feedback templates
        emotion_templates = {
            'Anger': [
                "Take a deep breath. Let's fix this step by step.",
                "Stay calm. This is a common mistake everyone makes.",
                "Don't worry, we can solve this together.",
                "It's frustrating, but you're learning! Let's check the {focus}."
            ],
            'Happiness': [
                "Great effort! Just one small fix needed.",
                "You're almost there! Just check the {focus}.",
                "Nice try! Almost perfect - just review the {focus}.",
                "Good work so far! Just need to fix the {focus}."
            ],
            'Calmness': [
                "Review the {focus} carefully.",
                "Check your {focus} syntax.",
                "Consider the proper {focus} usage.",
                "Verify the {focus} structure."
            ],
            'Sadness': [
                "Don't give up! Everyone struggles with {focus} at first.",
                "Keep trying! {focus} takes practice to master.",
                "It's okay to make mistakes. Focus on the {focus}.",
                "You're learning! {focus} will become easier with time."
            ],
            'Neutral': [
                "Check your {focus} syntax.",
                "Review the {focus} usage.",
                "Verify the {focus} structure.",
                "Examine the {focus} carefully."
            ]
        }

        prompt = f"""
            You are an expert SQL educator creating training data for an AI tutoring system.

            Database Schema:
            {schema_text}

            TASK: Create a realistic SQL query that contains EXACTLY this error:
            - Error Type: {error_type}
            - Error Subtype: {error_subtype}
            - Requirement: {specific_guidance}

            QUALITY REQUIREMENTS:
            1. Query MUST contain the specified error (this is critical for training)
            2. Make it look like something a student would actually write
            3. Use realistic column/table names from the provided schema
            4. Only include the specified error type, no additional errors
            5. Give correction hints in feedback_target that align with the target emotion
            6. Include simple, medium, and complex queries

            Target Emotion: {target_emotion}

            Provide ONLY valid JSON response:
            {{
                "query": "your SQL query with the specified error",
                "error_type": "{error_type}",
                "error_subtype": "{error_subtype}", 
                "emotion": "{target_emotion}",
                "feedback_target": "one educational hint sentence matching the emotion",
                "intended_learning_outcome": "specific learning objective for this error type"
            }}

            EXAMPLES:

            For syntax/misspelling:
            {{"query": "SELCT Name, Job_Role FROM Employees WHERE Division = 'Engineering';", "error_type": "syntax", "error_subtype": "misspelling", "emotion": "Happiness", "feedback_target": "Great effort! Just fix that small typo in SELECT.", "intended_learning_outcome": "Correct SQL keyword spelling"}}

            For schema/undefined column:
            {{"query": "SELECT ID, Name FROM Employees;", "error_type": "schema", "error_subtype": "undefined column", "emotion": "Calmness", "feedback_target": "Check the column names in the schema carefully.", "intended_learning_outcome": "Database schema awareness"}}

            For logic/aggregation misuse:
            {{"query": "SELECT Name, COUNT(*) FROM Employees;", "error_type": "logic", "error_subtype": "aggregation misuse", "emotion": "Sadness", "feedback_target": "Don't give up! GROUP BY is needed when mixing regular columns with aggregate functions.", "intended_learning_outcome": "GROUP BY with aggregate functions"}}
            """
        return prompt

    def generate_single_record(self, error_type: str, error_subtype: str, max_attempts: int = 3) -> Dict:
        """Generate a single record"""
        
        for attempt in range(max_attempts):
            self.generation_stats['total_attempts'] += 1
            
            try:
                prompt = self.create_detailed_prompt(error_type, error_subtype)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a SQL education expert. Always respond with valid JSON. Be precise about the error types requested."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    max_tokens=500
                )
                
                content = response.choices[0].message.content.strip()

                # Clean up content if wrapped in code block
                if content.startswith('```json'):
                    content = content[7:]
                elif content.endswith('```'):
                    content = content[:-3]
                elif content.startswith('```'):
                    content = content[3:]
    
                content = content.strip()
    
                # Extract JSON object from content
                brace_count = 0
                json_start = content.find('{')
                if json_start != -1:
                    for i, char in enumerate(content[json_start:], json_start):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                        
                        if brace_count == 0:
                            content = content[json_start:i+1]
                            break
    
                record = json.loads(content)
                
                # Basic completeness check only
                required_fields = ['query', 'error_type', 'error_subtype', 'emotion', 'feedback_target', 'intended_learning_outcome']
                if all(field in record and record[field] for field in required_fields):
                    self.generation_stats['successful_generations'] += 1
                    return record
            except Exception as e:
                self.generation_stats['api_failures'] += 1
                if attempt == 0:
                    print(f"API error for {error_type}/{error_subtype}: {str(e)[:50]}...")
                
                if attempt < max_attempts - 1:
                    time.sleep(1)
        
        # Use fallback generation
        self.generation_stats['fallback_used'] += 1
        print(f"Using fallback for {error_type}/{error_subtype}")
        return self.generate_fallback_record(error_type, error_subtype)
        
    def generate_fallback_record(self, error_type: str, error_subtype: str) -> Dict:
        """Fallback record generation"""
        
        # Predefined examples
        fallback_examples = {
            # Syntax errors
            ('syntax', 'misspelling'): {
                'queries': ["SELCT * FROM Employees;", "SELECT * FORM Employees;", "SELECT * FROM Employees WERE Name = 'John';"],
                'feedback': "Great effort! Just fix that small typo in the SQL keyword.",
                'outcome': "Correct SQL keyword spelling"
            },
            ('syntax', 'missing quotes'): {
                'queries': ["SELECT * FROM Employees WHERE Name = John;", "SELECT * FROM Robots WHERE Status = Active;"],
                'feedback': "Almost there! String values need quotes around them.",
                'outcome': "String literal syntax with quotes"
            },
            ('syntax', 'missing commas'): {
                'queries': ["SELECT Name Job_Role Division FROM Employees;", "SELECT Robot_ID Model Status FROM Robots;"],
                'feedback': "You're close! Just need commas between column names.",
                'outcome': "Column list punctuation in SELECT"
            },
            
            # Schema errors  
            ('schema', 'undefined column'): {
                'queries': ["SELECT ID FROM Employees;", "SELECT Name, Department FROM Robots;"],
                'feedback': "Check the column names in the schema carefully.",
                'outcome': "Database schema column awareness"
            },
            ('schema', 'undefined table'): {
                'queries': ["SELECT * FROM Worker;", "SELECT Name FROM Salary;"],
                'feedback': "Verify that the table name exists in the database.",
                'outcome': "Database schema table awareness"
            },
            
            # Logic errors
            ('logic', 'aggregation misuse'): {
                'queries': ["SELECT Name, COUNT(*) FROM Employees;", "SELECT Job_Role, SUM(Employee_ID) FROM Employees;"],
                'feedback': "When using aggregate functions, you need GROUP BY for non-aggregate columns.",
                'outcome': "GROUP BY with aggregate functions"
            },
            ('logic', 'incorrect JOIN usage'): {
                'queries': ["SELECT * FROM Employees JOIN Robots;", "SELECT * FROM Employees INNER JOIN Logs;"],
                'feedback': "JOIN clauses need ON conditions to specify how tables relate.",
                'outcome': "JOIN syntax and conditions"
            }
        }
        
        # Get example or create default
        example_key = (error_type, error_subtype)
        if example_key in fallback_examples:
            example = fallback_examples[example_key]
            query = random.choice(example['queries'])
            feedback = example['feedback']
            outcome = example['outcome']
        else:
            # Default fallback
            query = "SELECT * FROM Employees;"
            feedback = "Review the query syntax carefully."
            outcome = f"{error_type} error handling"
        
        emotion = random.choice(['Happiness', 'Calmness', 'Neutral'])
        
        return {
            'query': query,
            'error_type': error_type,
            'error_subtype': error_subtype,
            'emotion': emotion,
            'feedback_target': feedback,
            'intended_learning_outcome': outcome
        }

    def generate_balanced_dataset(self, total_records: int = 500) -> List[Dict]:
        """Generate dataset with balanced distribution"""
        
        # Calculate balanced distribution
        distribution = self.calculate_balanced_distribution(total_records)
        
        dataset = []
        current_record = 0
        
        print(f"\n Starting balanced generation of {total_records} records...")
        
        # Generate records according to distribution
        for error_type, subtypes in distribution.items():
            print(f"\n Generating {error_type} errors...")
            
            for error_subtype, count in subtypes.items():
                print(f"   → {error_subtype}: generating {count} records", end="")
                
                for i in range(count):
                    record = self.generate_single_record(error_type, error_subtype)
                    dataset.append(record)
                    current_record += 1
                    
                    # Progress display
                    if (current_record) % 25 == 0:
                        print(f"\n  Progress: {current_record}/{total_records} ({current_record/total_records*100:.1f}%)")
                        self.print_current_stats()
                    
                    # API rate limiting
                    time.sleep(0.15)
                
                print(f" - Complete")
        
        return dataset

    def print_current_stats(self):
        """Print current statistics"""
        stats = self.generation_stats
        if stats['total_attempts'] > 0:
            success_rate = stats['successful_generations'] / stats['total_attempts'] * 100
            print(f"    Current success rate: {success_rate:.1f}% (fallbacks: {stats['fallback_used']})")

    def save_dataset(self, dataset: List[Dict], filename: str = None):
        """Save dataset to CSV"""
        if filename is None:
            filename = f'balanced_sql_dataset_{len(dataset)}_records.csv'
        
        fieldnames = ['query', 'error_type', 'error_subtype', 'emotion', 'feedback_target', 'intended_learning_outcome']
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dataset)
        
        print(f"\n Dataset saved to: {filename}")
        return filename

    def print_final_statistics(self, dataset: List[Dict]):
        """Print final statistics"""
        stats = self.generation_stats
        
        print(f"\n" + "="*60)
        print(f" FINAL GENERATION REPORT")
        print(f"="*60)
        
        print(f"   Generation Statistics:")
        print(f"   Total API calls: {stats['total_attempts']}")
        print(f"   Successful generations: {stats['successful_generations']}")
        print(f"   API failures: {stats['api_failures']}")
        print(f"   Fallbacks used: {stats['fallback_used']}")
        if stats['total_attempts'] > 0:
            print(f"   Overall success rate: {stats['successful_generations']/stats['total_attempts']*100:.1f}%")
        
        # Distribution statistics
        error_counts = {}
        emotion_counts = {}
        subtype_counts = {}
        
        for record in dataset:
            # Error type distribution
            error_type = record['error_type']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            # Emotion distribution  
            emotion = record['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Subtype distribution
            subtype = f"{error_type}/{record['error_subtype']}"
            subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
        
        print(f"\n Distribution Analysis:")
        print(f"   Error type distribution:")
        for error_type, count in sorted(error_counts.items()):
            print(f"     {error_type}: {count} ({count/len(dataset)*100:.1f}%)")
        
        print(f"   Emotion distribution:")
        for emotion, count in sorted(emotion_counts.items()):
            print(f"     {emotion}: {count} ({count/len(dataset)*100:.1f}%)")
        
        print(f"\n Balance Check (top subtypes):")
        for subtype, count in sorted(subtype_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"     {subtype}: {count}")

def main():
    """Main function"""
    # Configuration
    API_KEY = os.getenv('OPENAI_API_KEY', 'api_key="replace_with_your_api_key"')  
    MODEL = "gpt-3.5-turbo"  
    NUM_RECORDS = 500
    
    print("  SQL Error Dataset Generator with Balanced Distribution")
    print(f"   Target: {NUM_RECORDS} records")
    print(f"   Model: {MODEL}")
    print(f"   Balance: Equal distribution across all subtypes")
    
    # Initialize generator
    generator = SQLDatasetGenerator(API_KEY, MODEL)
    
    # Generate balanced dataset
    dataset = generator.generate_balanced_dataset(NUM_RECORDS)
    
    # Save dataset
    filename = generator.save_dataset(dataset)
    
    # Print final report
    generator.print_final_statistics(dataset)
    
    # Show sample records
    print(f"\n Sample Records:")
    print("-" * 80)
    for i, record in enumerate(dataset[:3]):
        print(f"\nRecord {i+1} ({record['error_type']}/{record['error_subtype']}):")
        print(f"  Query: {record['query']}")
        print(f"  Emotion: {record['emotion']}")
        print(f"  Feedback: {record['feedback_target']}")
        print(f"  Learning: {record['intended_learning_outcome']}")

if __name__ == "__main__":
    main()