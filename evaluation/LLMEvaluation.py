import sys
from pathlib import Path
from compare_results import compare_results_f1
print("Importing SQLgenerator")
sys.path.insert(0, str(Path(__file__).parent.parent))
from SQLgenerator import explain_sql
print("Importing SQLEvaluator")
from SQLEvaluator import SQLComplexityEvaluator
print("Importing SematicScoring" )
from SematicScoring import calculate_similarity
print("Importing queryexecutor")
from queryexecutor import QueryExecutor
print("Importing pipeline")
from pipeline import stream_question_agent,build_graph
print("Importing the rest")
import json
import time
import pandas as pd
import gc
#--- to delte ----
import psutil
import os
#------------------

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return f"{process.memory_info().rss / (1024 * 1024):.2f} MB"

qe = QueryExecutor(max_rows=100000)

BASE_MODEL = "openai/gpt-4o-mini"
models = [
    "openai/gpt-4o-mini",
    "deepseek/deepseek-v3.2",
    "qwen/qwen-max",
    #"anthropic/claude-3-5-sonnet-20241022",
    #"arcee-ai/trinity-large-preview:free",
    #"stepfun/step-3.5-flash:free",
    #"z-ai/glm-4.5-air:free",
    #"nvidia/nemotron-3-nano-30b-a3b:free"
]

def evaluate_llm_performance(models, eval_dataset, qe):
    results = []
    evaluator = SQLComplexityEvaluator()

    for model in models:
        print(f"\n--- Starting Evaluation for Model: {model} ---")
        
        for case in eval_dataset:
            graph = build_graph(model)
            response = None
            gc.collect()
            print(f"Memory before case: {get_memory_usage()}")
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
                # 1. Start the timer
                start_time = time.perf_counter()

                # 2. Call the generator
                response = stream_question_agent(prompt, model)

                final_payload = None
                error_payload = None
                ttft = None  # Time to First Token

                # 3. Process the stream
                for event in response:
                    #print("Captured Events: " + event)
                    # Capture TTFT (First data event)
                    if ttft is None and event.startswith("data: "):
                        ttft = time.perf_counter() - start_time
                    
                    # Parse SSE format
                    if event.startswith("data: "):
                        try:
                            data = json.loads(event[6:])
                            # The 'done' event contains the payload with usage/sql
                            if data.get("type") == "done":
                                final_payload = data
                            elif data.get("type") == "error":
                                error_payload = data
                        except json.JSONDecodeError:
                            continue

                # 4. Calculate total latency after stream concludes
                latency = time.perf_counter() - start_time

                # 5. Extract results safely
                if error_payload:
                    row_data["Error Message"] = " | ".join(error_payload.get("reasons", ["Unknown Error"]))
                    generated_sql = "ERROR"
                if final_payload:
                    generated_sql = final_payload.get("final_sql", "")
                    input_tokens = final_payload.get("input_tokens", 0)
                    output_tokens = final_payload.get("output_tokens", 0)
                    total_tokens = final_payload.get("total_tokens", 0)
                    prompt_cost = final_payload.get("prompt_cost", 0.0)
                    completion_cost = final_payload.get("completion_cost", 0.0)
                    total_cost = final_payload.get("cost", 0.0)
                else:
                    # Handle the "Max loop reached" or crash scenario
                    generated_sql = "ERROR"
                    input_tokens = 0
                    output_tokens = 0
                    prompt_cost = 0.0
                    total_tokens = 0.0
                    completion_cost = 0.0
                    total_cost = 0.0
                
                # 2. Extract Token and Cost Data
                row_data["Generated SQL"] = generated_sql
                row_data["Latency (s)"] = round(latency, 3)
                row_data["Input Tokens"] = input_tokens
                row_data["Output Tokens"] = output_tokens
                row_data["Total Tokens"] = total_tokens
                row_data["Prompt Cost"] = prompt_cost
                row_data["Completion Cost"] = completion_cost
                row_data["Total Cost"] = total_cost

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
NCCS_Test_case = [
    {
        "Prompt": "What was the total number of patients diagnosed with colorectal cancer in 2020?",
        "Gold": "SELECT COUNT(*) AS count FROM condition_occurrence WHERE year(condition_start_date) = 2020;"
    },
    {
        "Prompt": "How many cases of colorectal cancer were classified as signet ring adenocarcinoma histological type?",
        "Gold": "SELECT COUNT(*) AS count FROM condition_occurrence WHERE LOWER(condition_source_value) LIKE '%signet ring%';"
    },
    {
        "Prompt": "What is the distribution of colorectal cancer patients by ethnicity, age, and gender?",
        "Gold": "SELECT gender_source_value, race_source_value, CASE WHEN YEAR(current_date) - year_of_birth < 50 THEN '<50' WHEN YEAR(current_date) - year_of_birth BETWEEN 50 AND 64 THEN '50-64' WHEN year(current_date) - year_of_birth BETWEEN 65 AND 74 THEN '65-74' ELSE '>75' END AS age_group, COUNT(distinct a.person_id) AS count FROM person a JOIN condition_occurrence b on a.person_id = b.person_id GROUP BY gender_source_value, race_source_value, age_group ORDER BY count DESC;"
    },
    {
        "Prompt": "What is the current staging distribution for colorectal cancer cases?",
        "Gold": "SELECT COUNT(*) AS count, value_as_concept_name AS cancer_stage FROM measurement_mutation WHERE lower(measurement_concept_name) like '%stage%' GROUP BY cancer_stage;"
    },
    {
        "Prompt": "What trends are observed in cancer staging over time?",
        "Gold": "SELECT YEAR(m.measurement_date) AS diagnosis_year, m.value_as_concept_name AS cancer_stage, COUNT(*) AS count FROM measurement_mutation m WHERE lower(measurement_concept_name) like '%stage%' GROUP BY diagnosis_year, cancer_stage;"
    },
    {
        "Prompt": "Which demographic groups (age, ethnicity, gender) are more likely to present with advanced-stage disease?",
        "Gold": "SELECT gender_source_value, race_source_value, CASE WHEN YEAR(current_date) - year_of_birth < 50 THEN '<50' WHEN YEAR(current_date) - year_of_birth BETWEEN 50 AND 64 THEN '50-64' WHEN year(current_date) - year_of_birth BETWEEN 65 AND 74 THEN '65-74' ELSE '>75' END AS age_group, COUNT(distinct a.person_id) AS count FROM person a LEFT JOIN measurement_mutation b on a.person_id = b.person_id WHERE measurement_concept_name like '%stage%' and value_as_concept_name like 'III%' or value_as_concept_name like 'IV%' GROUP BY gender_source_value, race_source_value, age_group ORDER BY count DESC;"
    },
    {
        "Prompt": "How many early-onset colorectal cancer cases (patients under 50 years old) were diagnosed in 2021?",
        "Gold": "SELECT COUNT(*) AS count FROM condition_occurrence a left join person b on a.person_id = b.person_id WHERE year(condition_start_date) = 2021 and year(current_date) - year_of_birth < 50;"
    },
    {
        "Prompt": "What percentage of colorectal cancer cases have KRAS, BRAF, or NRAS mutations?",
        "Gold": "WITH cohort AS (SELECT DISTINCT person_id FROM condition_occurrence), mutations AS (SELECT DISTINCT person_id FROM measurement_mutation LEFT JOIN cohort USING (person_id) WHERE (measurement_concept_name LIKE '%KRAS%' OR measurement_concept_name LIKE '%BRAF%' OR measurement_concept_name LIKE '%NRAS%') AND value_as_concept_name = 'mutation detected') SELECT ROUND(100.0 * (SELECT COUNT(*) FROM mutations) / (SELECT COUNT(*) FROM cohort), 2) AS mutation_percentage;"
    },
    {
        "Prompt": "How many colorectal cancer patients underwent surgical treatment in 2022?",
        "Gold": "SELECT COUNT(*) AS num_rows FROM procedure_occurrence LEFT JOIN condition_occurrence USING (person_id) WHERE YEAR(procedure_date) = 2022;"
    }
]

evaluation_data = evaluate_llm_performance(models, NCCS_Test_case,qe)
print(evaluation_data)
evaluation_data.to_excel("nccs_evaluation_results(NCCS_test_paid_V3).xlsx",index=False)