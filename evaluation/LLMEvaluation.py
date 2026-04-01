import sys
from datetime import datetime
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
from pipeline import stream_question_agent
print("Importing the rest")
import math
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
    "qwen3:8b"
    #"openai/gpt-4o-mini",
    #"deepseek/deepseek-v3.2",
    #"qwen/qwen-max",
    #"anthropic/claude-3-5-sonnet-20241022",
    #"arcee-ai/trinity-large-preview:free",
    #"stepfun/step-3.5-flash:free",
    #"z-ai/glm-4.5-air:free",
    #"nvidia/nemotron-3-nano-30b-a3b:free"
]
PENALTY_SCORE = 0.2

def evaluate_llm_performance(models, eval_dataset, qe):
    results = []
    evaluator = SQLComplexityEvaluator()

    for model in models:
        print(f"\n--- Starting Evaluation for Model: {model} ---")
        
        for case in eval_dataset:
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
                validation_tries = 0

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
                    validation_tries = final_payload.get("validation_tries", 0)
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
                row_data["Latency (s)"] = round(latency * (1 + ((validation_tries - 1) * PENALTY_SCORE)), 3)
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

    evaluation_df = pd.DataFrame(results)
    evaluation_df["Log Transformed Tokens"] = evaluation_df["Total Tokens"].apply(lambda x: round(math.log(x + 1), 3)) # Log transform for better scaling
    columns_to_normalize = ["Latency (s)", "Complexity Score", "Log Transformed Tokens", "F1 Score", "Semantic Score"]
    for column in columns_to_normalize:
        evaluation_df[f"Normalized {column}"] = normalize_score(column, evaluation_df)
    evaluation_df["Efficiency Score"] = 0.6 * evaluation_df["Normalized F1 Score"] + 0.1 * evaluation_df["Normalized Latency (s)"] + 0.1 * evaluation_df["Normalized Log Transformed Tokens"] + 0.2 * evaluation_df["Normalized Complexity Score"]
    return evaluation_df

def normalize_score(column, df):
    min_max_columns = ["F1 Score", "Semantic Score"] # Higher value is better
    filtered_df = df[df["Generated SQL"] != "ERROR"] # Only consider successful generations for normalization
    if filtered_df.empty:
        return df[column].apply(lambda x: 0.5) # If no successful generations, assign 0.5 to all
    min_score = filtered_df[column].min()
    max_score = filtered_df[column].max()
    if max_score - min_score == 0:
        return df[column].apply(lambda x: 0.5) # If all scores are the same, assign 0.5
    elif column in min_max_columns:
        return df[column].apply(lambda x: (x - min_score) / (max_score - min_score))
    else: # reverse normalization applied (lower is better)
        return df[column].apply(lambda x: (max_score - x) / (max_score - min_score))

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
        "Gold": "SELECT COUNT(DISTINCT person_id) AS count FROM condition_occurrence WHERE (ICD10 ilike 'C18%' or ICD10 ilike 'C19' or ICD10 ilike 'C20') and YEAR(condition_start_date) = 2020"
    },
    {
        "Prompt": "How many cases of colorectal cancer were classified as signet ring adenocarcinoma histological type?",
        "Gold": "SELECT COUNT(person_id) AS count FROM (SELECT *, trim(split(condition_source_value, '||')[4]) as Histo2 FROM condition_occurrence) where Histo2 ilike '%Signet ring%'"
    },
    {
        "Prompt": "What is the patient counts for colorectal cancer by ethnic group?",
        "Gold": "SELECT COUNT(*) as n, race_source_value FROM person GROUP BY race_source_value ORDER BY n DESC"
    },
    {
        "Prompt": "Show me the number of colorectal cancer cases in each 5‑year age‑at‑diagnosis group.",
        "Gold": "WITH crc AS (SELECT b.person_id, b.condition_start_date AS dx_date, b.ICD10 FROM condition_occurrence b WHERE b.ICD10 LIKE 'C18%' OR b.ICD10 LIKE 'C19' OR b.ICD10 LIKE 'C20'), ages AS (SELECT c.person_id, c.dx_date, EXTRACT(YEAR FROM c.dx_date) - a.year_of_birth - CASE WHEN EXTRACT(MONTH FROM c.dx_date) <= a.month_of_birth THEN 1 ELSE 0 END AS age FROM crc c JOIN person a USING(person_id)), bands AS (SELECT *, CAST(FLOOR(age / 5) * 5 AS INT) AS age_lo, CAST(FLOOR(age / 5) * 5 + 4 AS INT) AS age_hi FROM ages) SELECT CONCAT(CAST(age_lo AS STRING), '-', CAST(age_hi AS STRING)) AS age_group, COUNT(*) AS n_cases FROM bands GROUP BY age_lo, age_hi ORDER BY age_lo;"
    },
    {
        "Prompt": "What is the breakdown of colorectal cancer cases by stage? [tips: using p group stage, if null, then use c group stage]",
        "Gold": "WITH colorectal_staging AS (SELECT person_id, measurement_concept_name, value_source_value AS stage, measurement_date, ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY CASE WHEN measurement_concept_name = 'TNM Path Stage Group' THEN 1 WHEN measurement_concept_name = 'TNM Clin Stage Group' THEN 2 ELSE 3 END, measurement_date DESC) AS rn FROM measurement_mutation WHERE measurement_concept_name IN ('TNM Path Stage Group', 'TNM Clin Stage Group') AND person_id IN (SELECT DISTINCT person_id FROM condition_occurrence WHERE ICD10 LIKE 'C18%' OR ICD10 LIKE 'C19' OR ICD10 LIKE 'C20')), final_stages AS (SELECT person_id, CASE WHEN TRIM(UPPER(stage)) LIKE 'IV%' THEN 'IV' WHEN TRIM(UPPER(stage)) LIKE 'III%' THEN 'III' WHEN TRIM(UPPER(stage)) LIKE 'II%' THEN 'II' WHEN TRIM(UPPER(stage)) LIKE 'I%' THEN 'I' ELSE 'Unknown' END AS cleaned_stage, measurement_concept_name AS stage_type FROM colorectal_staging WHERE rn = 1) SELECT cleaned_stage AS stage, COUNT(*) AS case_count FROM final_stages WHERE cleaned_stage != 'Unknown' GROUP BY cleaned_stage ORDER BY cleaned_stage;"
    },
    {
        "Prompt": "What are the trends for each colorectal cancer staging group across diagnosis years?",
        "Gold": "WITH colorectal_staging AS (SELECT person_id, measurement_concept_name, value_source_value AS stage, measurement_date, ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY CASE WHEN measurement_concept_name = 'TNM Path Stage Group' THEN 1 WHEN measurement_concept_name = 'TNM Clin Stage Group' THEN 2 ELSE 3 END, measurement_date DESC) AS rn FROM measurement_mutation WHERE measurement_concept_name IN ('TNM Path Stage Group', 'TNM Clin Stage Group') AND person_id IN (SELECT DISTINCT person_id FROM condition_occurrence WHERE ICD10 LIKE 'C18%' OR ICD10 LIKE 'C19' OR ICD10 LIKE 'C20')), final_stages AS (SELECT person_id, measurement_date, CASE WHEN TRIM(UPPER(stage)) LIKE 'IV%' THEN 'IV' WHEN TRIM(UPPER(stage)) LIKE 'III%' THEN 'III' WHEN TRIM(UPPER(stage)) LIKE 'II%' THEN 'II' WHEN TRIM(UPPER(stage)) LIKE 'I%' THEN 'I' ELSE 'Unknown' END AS cleaned_stage, measurement_concept_name AS stage_type FROM colorectal_staging WHERE rn = 1) SELECT YEAR(measurement_date) AS dx_year, cleaned_stage AS stage, COUNT(*) AS case_count FROM final_stages WHERE cleaned_stage != 'Unknown' GROUP BY cleaned_stage, dx_year ORDER BY dx_year, cleaned_stage;"
    },
    {
        "Prompt": "Which age‑at‑diagnosis groups (in 5‑year intervals) show a highest number of patients diagnosed with stage IV colorectal cancer?",
        "Gold": "WITH crc AS (SELECT b.person_id, b.condition_start_date AS dx_date, b.ICD10 FROM condition_occurrence b WHERE b.ICD10 LIKE 'C18%' OR b.ICD10 LIKE 'C19' OR b.ICD10 LIKE 'C20'), ages AS (SELECT c.person_id, c.dx_date, EXTRACT(YEAR FROM c.dx_date) - a.year_of_birth - CASE WHEN EXTRACT(MONTH FROM c.dx_date) <= a.month_of_birth THEN 1 ELSE 0 END AS age FROM crc c JOIN person a USING(person_id)), age_bands AS (SELECT person_id, age, CAST(FLOOR(age / 5) * 5 AS INT) AS age_lo, CAST(FLOOR(age / 5) * 5 + 4 AS INT) AS age_hi FROM ages), colorectal_staging AS (SELECT person_id, measurement_concept_name, value_source_value as stage, measurement_date, ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY CASE WHEN measurement_concept_name = 'TNM Path Stage Group' THEN 1 WHEN measurement_concept_name = 'TNM Clin Stage Group' THEN 2 ELSE 3 END, measurement_date DESC) as rn FROM measurement_mutation WHERE measurement_concept_name IN ('TNM Path Stage Group', 'TNM Clin Stage Group') AND person_id IN (SELECT DISTINCT person_id FROM condition_occurrence WHERE ICD10 LIKE 'C18%' OR ICD10 LIKE 'C19' OR ICD10 LIKE 'C20')), stage_iv_patients AS (SELECT person_id FROM colorectal_staging WHERE rn = 1 AND TRIM(UPPER(stage)) LIKE 'IV%') SELECT CONCAT(CAST(age_lo AS STRING), '-', CAST(age_hi AS STRING)) AS age_group, COUNT(*) AS stage_iv_cases FROM age_bands ab JOIN stage_iv_patients siv ON ab.person_id = siv.person_id GROUP BY age_lo, age_hi ORDER BY stage_iv_cases DESC, age_lo LIMIT 1"
    },
    {
        "Prompt": "Show me the number of early‑onset colorectal cancer cases (age at diagnosis ≤ 49) diagnosed in 2021.",
        "Gold": "WITH crc AS (SELECT b.person_id, b.condition_start_date AS dx_date, b.ICD10 FROM condition_occurrence b WHERE (b.ICD10 LIKE 'C18%' OR b.ICD10 LIKE 'C19' OR b.ICD10 LIKE 'C20') AND EXTRACT(YEAR FROM b.condition_start_date) = 2021), ages AS (SELECT c.person_id, c.dx_date, EXTRACT(YEAR FROM c.dx_date) - a.year_of_birth - CASE WHEN EXTRACT(MONTH FROM c.dx_date) <= a.month_of_birth THEN 1 ELSE 0 END AS age FROM crc c JOIN person a USING(person_id)) SELECT COUNT(*) AS early_onset_cases_2021 FROM ages WHERE age <= 49;"
    },
    {
        "Prompt": "Among patients who were tested, what proportion of colorectal cancer cases show KRAS, BRAF, or NRAS mutations? [noted there are some missing values, please include those in analysis]",
        "Gold": "WITH colorectal_patients AS (SELECT DISTINCT person_id FROM condition_occurrence WHERE (ICD10 ILIKE 'C18%' OR ICD10 ILIKE 'C19' OR ICD10 ILIKE 'C20')), mutation_summary AS (SELECT m.person_id, SUM(CASE WHEN m.measurement_concept_name = 'KRAS Mutation Conclusion' AND m.value_source_value = 'mutation detected' THEN 1 ELSE 0 END) AS kras_positive, SUM(CASE WHEN m.measurement_concept_name = 'BRAF Mutation Conclusion' AND m.value_source_value = 'mutation detected' THEN 1 ELSE 0 END) AS braf_positive, SUM(CASE WHEN m.measurement_concept_name = 'NRAS Mutation Conclusion' AND m.value_source_value = 'mutation detected' THEN 1 ELSE 0 END) AS nras_positive, COUNT(DISTINCT m.measurement_concept_name) AS tests_performed FROM measurement_mutation m INNER JOIN colorectal_patients cp ON m.person_id = cp.person_id WHERE m.measurement_concept_name IN ('KRAS Mutation Conclusion', 'BRAF Mutation Conclusion', 'NRAS Mutation Conclusion') GROUP BY m.person_id) SELECT COUNT(*) AS total_tested_patients, SUM(CASE WHEN kras_positive > 0 THEN 1 ELSE 0 END) AS patients_with_kras, ROUND((SUM(CASE WHEN kras_positive > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) AS kras_percentage, SUM(CASE WHEN braf_positive > 0 THEN 1 ELSE 0 END) AS patients_with_braf, ROUND((SUM(CASE WHEN braf_positive > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) AS braf_percentage, SUM(CASE WHEN nras_positive > 0 THEN 1 ELSE 0 END) AS patients_with_nras, ROUND((SUM(CASE WHEN nras_positive > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) AS nras_percentage FROM mutation_summary;"
    },
    {
        "Prompt": "How many colorectal cancer patients with KRAS mutations are female?",
        "Gold": "WITH colorectal_patients AS (SELECT DISTINCT person_id FROM condition_occurrence WHERE (ICD10 ILIKE 'C18%' OR ICD10 ILIKE 'C19' OR ICD10 ILIKE 'C20')), kras_positive_patients AS (SELECT DISTINCT m.person_id FROM measurement_mutation m INNER JOIN colorectal_patients cp ON m.person_id = cp.person_id WHERE m.measurement_concept_name = 'KRAS Mutation Conclusion' AND m.value_source_value = 'mutation detected') SELECT COUNT(DISTINCT kpp.person_id) AS female_kras_positive_patients FROM kras_positive_patients kpp INNER JOIN person d ON kpp.person_id = d.person_id WHERE LOWER(d.gender_source_value) = 'female';"
    },
    {
        "Prompt": "How many patients with KRAS wild type received anti‑EGFR therapy (panitumumab or cetuximab)? [tips: wild type = no mutation; noted there are some missing values in drug_exposure_start_date, please include those missing dates as well, assume the drug was given]",
        "Gold": "WITH colorectal_patients AS (SELECT DISTINCT person_id FROM condition_occurrence WHERE (ICD10 ILIKE 'C18%' OR ICD10 ILIKE 'C19' OR ICD10 ILIKE 'C20')) SELECT COUNT(DISTINCT m.person_id) AS kras_wild_type_with_anti_egfr FROM measurement_mutation m INNER JOIN colorectal_patients cp ON m.person_id = cp.person_id INNER JOIN drug_exposure_cancerdrugs d ON m.person_id = d.person_id WHERE m.measurement_concept_name = 'KRAS Mutation Conclusion' AND m.value_source_value = 'no mutation detected' AND (LOWER(d.drug_source_value) ILIKE '%panitumumab%' OR LOWER(d.drug_source_value) ILIKE '%cetuximab%');"
    },
    {
        "Prompt": "Show me the number of stage IV Colorectal cancer patients who had a liver resection. [keywords: Liver, Lobectomy, Wedge/Local]",
        "Gold": "WITH colorectal_staging AS (SELECT person_id, measurement_concept_name, value_source_value AS stage, measurement_date, ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY CASE WHEN measurement_concept_name = 'TNM Path Stage Group' THEN 1 WHEN measurement_concept_name = 'TNM Clin Stage Group' THEN 2 ELSE 3 END, measurement_date DESC) AS rn FROM measurement_mutation WHERE measurement_concept_name IN ('TNM Path Stage Group', 'TNM Clin Stage Group') AND person_id IN (SELECT DISTINCT person_id FROM condition_occurrence WHERE ICD10 LIKE 'C18%' OR ICD10 LIKE 'C19' OR ICD10 LIKE 'C20')) SELECT COUNT(DISTINCT cs.person_id) AS stage_iv_with_liver_resection FROM colorectal_staging cs INNER JOIN procedure_occurrence s ON cs.person_id = s.person_id WHERE cs.rn = 1 AND TRIM(UPPER(cs.stage)) LIKE 'IV%' AND ((s.procedure_source_value ILIKE '%Liver%' AND s.procedure_source_value ILIKE '%LOBECTOMY%') OR s.procedure_source_value ILIKE '%WEDGE/LOCAL%');"
    },
    {
        "Prompt": "How many cancer types and patients are in this dataset?",
        "Gold": "SELECT ICD10, COUNT(DISTINCT person_id) as distinct_patients FROM condition_occurrence GROUP BY ICD10 ORDER BY distinct_patients DESC"
    },
    {
        "Prompt": "What are the total deaths and their proportion of colorectal cancer for the year 2010-2020?",
        "Gold": "WITH colorectal_patients_2010_2020 AS (SELECT DISTINCT person_id, condition_start_date FROM condition_occurrence WHERE (ICD10 ILIKE 'C18%' OR ICD10 ILIKE 'C19' OR ICD10 ILIKE 'C20') AND YEAR(condition_start_date) BETWEEN 2010 AND 2020) SELECT COUNT(DISTINCT cp.person_id) AS total_colorectal_patients_2010_2020, COUNT(DISTINCT d.person_id) AS total_deaths_colorectal_patients_2010_2020, ROUND((COUNT(DISTINCT d.person_id) * 100.0 / COUNT(DISTINCT cp.person_id)), 2) AS death_proportion_percentage FROM colorectal_patients_2010_2020 cp LEFT JOIN death d ON cp.person_id = d.person_id;"
    }
]

evaluation_data = evaluate_llm_performance(models, NCCS_Test_case,qe)
print(evaluation_data)


current_dir = Path(__file__).resolve().parent

project_root = current_dir.parent
eval_dir = project_root / "eval_files"
eval_dir.mkdir(parents=True, exist_ok=True)

# Generate a timestamp string (e.g., 20260329_2355)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
file_path = eval_dir / f"nccs_evaluation_results_{timestamp}.csv"
evaluation_data.to_csv(file_path, index=False)