import sys
from pathlib import Path
from compare_results import compare_results_f1
sys.path.insert(0, str(Path(__file__).parent.parent))
from SQLgenerator import generate_sql_from_nl,explain_sql
from SQLEvaluator import SQLComplexityEvaluator
from SematicScoring import calculate_similarity
from queryexecutor import QueryExecutor
import time
import pandas as pd

qe = QueryExecutor(max_rows=100000)

BASE_MODEL = "openai/gpt-4o-mini"
models = [
    #"openai/gpt-4o-mini",
    #"deepseek/deepseek-v3.2",
    #"qwen/qwen-max",
    "anthropic/claude-3-5-sonnet-20241022"
]

def evaluate_llm_performance(models, eval_dataset, qe):
    results = []
    evaluator = SQLComplexityEvaluator()

    for model in models:
        print(f"\n--- Starting Evaluation for Model: {model} ---")
        
        for case in eval_dataset:
            prompt = case["Prompt"]
            gold_sql = case["Gold"]
            print(f"Testing Prompt: {prompt[:50]}...")
            
            # Default values in case of failure
            row_data = {
                "Model": model,
                "Prompt": prompt,
                "Gold SQL": gold_sql,
                "Generated SQL": "ERROR",
                "Generated Explanation": "ERROR",
                "Complexity Score": None,
                "Complexity Level": "N/A",
                "Complexity Description": "N/A",
                "Latency (s)": 0,
                "Input Tokens": 0,
                "Output Tokens": 0,
                "Total Tokens": 0,
                "Prompt Cost": 0.0,
                "Completion Cost": 0.0,
                "Total Cost": 0.0,
                "Precision": 0,
                "Recall": 0,
                "F1 Score": 0,
                "Semantic Score": 0,
                "Error Message": None # Track what actually went wrong
            }

            try:
                # 1. Measure Latency and SQL Generation
                start_time = time.perf_counter()
                response = generate_sql_from_nl(prompt, model)
                latency = time.perf_counter() - start_time
                
                generated_sql = response.get("sql", "")
                usage = response.get("usage", {})
                cost_info = response.get("cost", {})
                
                # 2. Extract Token and Cost Data
                row_data["Generated SQL"] = generated_sql
                row_data["Latency (s)"] = round(latency, 3)
                row_data["Input Tokens"] = usage.get("input_tokens", 0)
                row_data["Output Tokens"] = usage.get("output_tokens", 0)
                row_data["Total Tokens"] = usage.get("total_tokens", 0)
                row_data["Prompt Cost"] = float(cost_info.get("prompt_cost", 0.0))
                row_data["Completion Cost"] = float(cost_info.get("completion_cost", 0.0))
                row_data["Total Cost"] = float(cost_info.get("total_cost", 0.0))

                # 3. Explanation and Semantic Score
                # Note: explain_sql should also have its own try-catch or be safe
                generated_explanation = explain_sql(generated_sql, model)
                row_data["Generated Explanation"] = generated_explanation
                row_data["Semantic Score"] = calculate_similarity(prompt, generated_explanation)

                # 4. Execution Metrics
                precision, recall, f1 = compare_results_f1(qe, gold_sql, generated_sql, debug=False)
                row_data["Precision"] = precision
                row_data["Recall"] = recall
                row_data["F1 Score"] = f1

                # 5. Complexity
                complexity = evaluator.rate_query(generated_sql)
                row_data["Complexity Score"] = complexity.get('score')
                row_data["Complexity Level"] = complexity.get('level')
                row_data["Complexity Description"] = complexity.get('description')

            except Exception as e:
                print(f"!!! Error on prompt: {prompt[:30]} | Error: {str(e)}")
                row_data["Error Message"] = str(e)
            
            finally:
                # 6. Record the data (Success or Error)
                results.append(row_data)

    return pd.DataFrame(results)


test_cases = [
    {
        "Prompt": "How many deaths occurred in 2020?",
        "Gold": "SELECT COUNT(*) AS death_count FROM death WHERE YEAR(death_date) = 2020;"
    },
    {
        "Prompt": "List the top 5 races based on number of person records. Order them in descending order of number of records.",
        "Gold": "SELECT race_source_value, COUNT(*) AS race_count FROM person GROUP BY race_source_value ORDER BY race_count DESC LIMIT 5;"
    },
    {
        "Prompt": "How many patients of each gender were diagnosed with ICD10 code C18.7 in 2021?",
        "Gold": "SELECT gender_source_value, COUNT(*) AS gender_count FROM person LEFT JOIN condition_occurrence USING (person_id) WHERE ICD10 = 'C18.7' AND YEAR(condition_start_date) = 2021 GROUP BY gender_source_value;"
    },
    {
        "Prompt": "Count the number of drug exposure events that started and ended in the year 2020.",
        "Gold": "SELECT COUNT(*) AS count FROM drug_exposure_cancerdrugs WHERE YEAR(drug_exposure_start_date) = 2020 AND YEAR(drug_exposure_end_date) = 2020;"
    },
    {
        "Prompt": "How many conditions or diagnoses are colon related?",
        "Gold": "SELECT COUNT(*) AS count FROM condition_occurrence WHERE condition_source_value LIKE '%colon%';"
    },
    {
        "Prompt": "How many male patients had conditions or diagnosis concerning the rectum?",
        "Gold": "SELECT COUNT(*) AS male_count FROM person JOIN condition_occurrence USING (person_id) WHERE person.gender_source_value = 'male' AND condition_occurrence.condition_source_value LIKE '%rectum%';"
    },
    {
        "Prompt": "How many active seniors aged 55 and above were prescribed with oxaliplatin?",
        "Gold": "SELECT COUNT(*) AS total_records FROM drug_exposure_cancerdrugs LEFT JOIN person USING (person_id) LEFT JOIN death USING (person_id) WHERE death_date IS NOT NULL AND YEAR(NOW()) - year_of_birth >= 55;"
    },
    {
        "Prompt": "What is the most common cause of death?",
        "Gold": "SELECT cause_source_value, COUNT(*) AS death_count FROM death GROUP BY cause_source_value ORDER BY death_count DESC LIMIT 1;"
    },
    {
        "Prompt": "Consolidate number of measurement, procedure and drug exposure events respectively.",
        "Gold": "SELECT 'measurement' AS event_type, COUNT(*) AS event_count FROM measurement_mutation UNION ALL SELECT 'procedure' AS event_type, COUNT(*) AS event_count FROM procedure_occurrence UNION ALL SELECT 'drug_exposure' AS event_type, COUNT(*) AS event_count FROM drug_exposure_cancerdrugs;"
    },
    {
        "Prompt": "Create a table reporting the number of procedure and drug exposure events respectively in each institution.",
        "Gold": "SELECT t1.institution, t1.drug_exposure_events, COALESCE(t2.procedure_occurrence_events, 0) AS procedure_occurrence_events FROM (SELECT institution, COUNT(*) AS drug_exposure_events FROM drug_exposure_cancerdrugs GROUP BY institution) t1 LEFT JOIN (SELECT institution, COUNT(*) AS procedure_occurrence_events FROM procedure_occurrence GROUP BY institution) t2 USING (institution);"
    },
    {
        "Prompt": "Generate a racial demographic of active patients.",
        "Gold": "SELECT race_source_value, COUNT(*) AS count FROM person LEFT JOIN death USING (person_id) WHERE death.person_id IS NULL GROUP BY race_source_value;"
    },
    {
        "Prompt": "Rank the causes of death based on number of occurrences.",
        "Gold": "SELECT cause_source_value, DENSE_RANK() OVER (ORDER BY COUNT(*) DESC) AS rank FROM death GROUP BY cause_source_value ORDER BY rank;"
    },
    {
        "Prompt": "How many deaths were associated with drug exposure events?",
        "Gold": "SELECT COUNT(*) AS num_of_deaths FROM drug_exposure_cancerdrugs LEFT JOIN death USING (person_id) WHERE death.person_id IS NOT NULL;"
    },
    {
        "Prompt": "How many of the drugs prescribed are capecitabine?",
        "Gold": "SELECT COUNT(*) AS count FROM drug_exposure_cancerdrugs WHERE LOWER(drug_source_value) LIKE '%capecitabine%';"
    },
    {
        "Prompt": "What are the different types of measurements used in the OMOP Common Data Model?",
        "Gold": "SELECT DISTINCT measurement_concept_name FROM measurement_mutation;"
    },
    {
        "Prompt": "Out of all the mutation measurements, how many of them did not detect mutations?",
        "Gold": "SELECT COUNT(*) AS count FROM measurement_mutation WHERE LOWER(measurement_concept_name) LIKE '%mutation%' AND LOWER(value_source_value) LIKE '%no mutation detected%';"
    },
    {
        "Prompt": "How many procedures in 2020 lasted for longer than a week?",
        "Gold": "SELECT COUNT(*) AS count FROM procedure_occurrence WHERE DATEDIFF('day', STRPTIME(procedure_datetime, '%d/%m/%Y %H:%M'), STRPTIME(procedure_end_datetime, '%d/%m/%Y %H:%M')) > 7 AND YEAR(STRPTIME(procedure_datetime, '%d/%m/%Y %H:%M')) = 2020;"
    },
    {
        "Prompt": "How many drug exposure events lasted for longer than 2 weeks?",
        "Gold": "SELECT COUNT(*) AS count FROM drug_exposure_cancerdrugs WHERE DATEDIFF('week', drug_exposure_start_date, drug_exposure_end_date) > 2;"
    },
    {
        "Prompt": "How many different types of diagnoses have been recorded under the ICD10 classification?",
        "Gold": "SELECT COUNT(DISTINCT ICD10) AS count FROM condition_occurrence;"
    },
    {
        "Prompt": "What percentage of drug exposure events recorded in 2019 involve inpatients?",
        "Gold": "SELECT ROUND(inpatient_count / NULLIF(total_count, 0) * 100, 2) AS inpatient_percentage FROM (SELECT SUM(CASE WHEN LOWER(case_type) = 'inpatient' THEN 1 ELSE 0 END) AS inpatient_count, COUNT(*) AS total_count FROM drug_exposure_cancerdrugs WHERE YEAR(drug_exposure_start_date) = 2019 AND YEAR(drug_exposure_end_date) = 2019);"
    }
]

evaluation_data = evaluate_llm_performance(models, test_cases,qe)
print(evaluation_data)
evaluation_data.to_excel("nccs_evaluation_results.xlsx",index=False)