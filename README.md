# Data Preprocessing Pipeline on Azure - Lab 3

Name: Arlene Riona

Student ID: 60304739

#

**Overview**

In this lab, we implemented the data preparation and feature engineering stage of the data science pipeline using Azure cloud technologies. The objective was to transform raw Amazon Electronics review data into a structured, analytics-ready dataset that can support downstream analysis, visualization, and machine learning tasks.

The workflow followed a Lakehouse Medallion Architecture, where data is progressively refined:

* Bronze Layer – Raw JSON data stored in Azure Data Lake Storage Gen2
* Silver Layer – Cleaned, validated, and structured Parquet data
* Gold Layer – Curated feature dataset ready for analytics and modeling

Using Azure Databricks for distributed data processing and Azure Data Lake Storage for scalable storage, we built an end-to-end preprocessing pipeline that performs data cleaning, validation, enrichment with metadata, and feature selection.

By the end of this lab, we produced a reliable and analysis-ready Gold dataset and orchestrated the workflow using Databricks Jobs, demonstrating how real-world data engineering pipelines prepare data before the modeling and deployment stages of the data science lifecycle.

#

**Technologies Used**

* Azure Data Lake Storage Gen2 (Data storage)
* Azure Databricks (Data processing environment)
* Databricks Notebooks (ETL logic)
* Databricks Jobs (Pipeline orchestration)
* Parquet format (Optimized storage for analytics)
* Python (PySpark DataFrame transformations)
* Matplotlib and Pandas (Data visualization)

#
**Steps Done**

**Step 1 — Set Up Azure Databricks Workspace**


We created an Azure Databricks workspace in the same region as the storage account. This workspace provides a managed environment for running data transformation notebooks.


#
**Step 2 — Create a Databricks Cluster**


A cluster was created to provide the compute resources required for running the notebooks.
Cluster configuration:
* Databricks Runtime: Long-Term Support (LTS)
* Worker nodes: Auto-scaling enabled
* Auto-termination enabled to control costs
This cluster executes all data transformation steps.


<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/4ba5d55e-5d21-45df-b436-208e8bf177ee" />
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/42f49ed9-de49-4549-9b48-37baef381634" />

#
**Step 3 — Notebook 1: Load and Clean Reviews Data**


In the first notebook, we prepared the review dataset stored in the Silver layer.

Steps performed:
1. Connected Databricks to Azure Data Lake Storage Gen2 using storage account credentials.
2. Loaded the reviews dataset stored in Parquet format from the processed container.
3. Applied cleaning and validation rules:
    * Removed rows missing critical fields such as asin, reviewerID, or overall
    * Ensured ratings were between 1 and 5
    * Trimmed review text and removed very short or empty reviews
4. Saved the cleaned dataset back to the Silver layer.

This step ensured that only valid and consistent data would move forward in the pipeline.


<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/4680805b-b26d-47da-9b9f-53bcc927f907" />
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/91e38a3b-5f99-4518-9f82-6b2ba6fc0948" />
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/f8fcfc14-0276-4360-a2fe-640ed322be82" />
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/bfe5344c-23d3-4ae9-ab7f-99fb6389ef6a" />


#
**Step 4 — Notebook 2: Enrich Reviews with Product Metadata**


The second notebook added product details to each review.

Steps performed:
1. Loaded the cleaned reviews from the Silver layer.
2. Loaded product metadata from the Bronze layer JSON file.
3. Selected only the required metadata fields: asin, title, brand, and price.
4. Joined reviews with metadata using the asin key.
5. Wrote the enriched dataset back to the Silver layer.

This step enhanced the review data by adding product information needed for analysis.


<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/38c67147-4e96-476e-ae8e-3f94afcd877d" />
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/fc939a87-3a9c-4add-b351-a9b700dfa593" />
<img width="550" height="300" alt="image" src="https://github.com/user-attachments/assets/6119526d-bbe0-48e1-8e58-5445d44eecac" />
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/1fae83b9-9324-4664-aa18-c5a2d71fa57a" />
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/b905b369-c742-4c39-b8f7-1372123d74d3" />

#
**Step 5 — Notebook 3: Create Gold Curated Dataset**


The third notebook created the final curated dataset for analysis.

Steps performed:

1. Loaded the enriched reviews dataset.
2. Selected the final set of columns required for analytics:
    * Product information: asin, title, brand, price
    * Review information: reviewerID, overall, summary, reviewText, helpful
    * Time information: reviewTime, review_year
3. Saved the final dataset in Parquet format to the Gold layer under curated/features_v1/.

This dataset is now structured and ready for reporting, visualization, or machine learning.

<img width="450" height="450" alt="image" src="https://github.com/user-attachments/assets/5dabe4e5-074d-4333-9ebb-7d4b57a36258" />
<img width="450" height="450" alt="image" src="https://github.com/user-attachments/assets/b9697fa9-e579-46fd-958b-0669306be39f" />
<img width="450" height="450" alt="image" src="https://github.com/user-attachments/assets/934db428-4f31-4e98-9794-0dcd668393a5" />
<img width="450" height="450" alt="image" src="https://github.com/user-attachments/assets/0c9d6b6e-842f-4ff7-9c4d-99cad2d39dbf" />

#
**Step 6 — Pipeline Orchestration with Databricks Jobs**


To automate the process, we created a Databricks Job that runs the notebooks sequentially.

Pipeline structure:

1. Load and clean reviews
2. Enrich with metadata
3. Write Gold dataset

Each task depends on the previous one, ensuring the correct execution order. The job can be triggered manually or scheduled.


<img width="450" height="450" alt="image" src="https://github.com/user-attachments/assets/06700c6a-44af-46bd-b1dd-14b2cde6fa80" />
<img width="450" height="450" alt="image" src="https://github.com/user-attachments/assets/47170b2b-c525-408d-9d51-be68d6b5baee" />
<img width="450" height="450" alt="image" src="https://github.com/user-attachments/assets/babca64b-7835-4bbc-9f10-9be2d348356a" />
<img width="450" height="450" alt="image" src="https://github.com/user-attachments/assets/7effc866-3252-4ec9-a15f-af4c26362fd9" />
<img width="450" height="450" alt="image" src="https://github.com/user-attachments/assets/f80928d4-8185-40a9-9c42-808a14e45b45" />
<img width="450" height="450" alt="image" src="https://github.com/user-attachments/assets/c936c66e-2af0-4d14-a1e8-70fa428c420d" />
<img width="450" height="450" alt="image" src="https://github.com/user-attachments/assets/dca04c39-bfce-4adc-9b33-4165ec5856f7" />
<img width="450" height="450" alt="image" src="https://github.com/user-attachments/assets/4cd23a84-9f6a-4330-9298-4630afd576ea" />

#
**Step 7 — Data Visualization**


Using the Gold dataset, we created visualizations to analyze trends in customer feedback.

Visualizations created:

1. Average Rating Over Time


This chart shows how customer ratings changed over the years. Ratings declined in the early years and then gradually increased and stabilized above 4, indicating improved customer satisfaction over time.

2. Distribution of Ratings


This chart shows that most reviews are 4 and 5 stars, while low ratings are much less common. This indicates overall positive customer feedback.


<img width="450" height="450" alt="image" src="https://github.com/user-attachments/assets/fa17f9d2-008f-4b94-aa9a-800d76744ea9" />
<img width="450" height="450" alt="image" src="https://github.com/user-attachments/assets/893a8137-3ed7-4d46-98a9-9e217c4d79ea" />
<img width="450" height="450" alt="image" src="https://github.com/user-attachments/assets/2a2e6912-71c9-464b-a279-a0a87423d31a" />


#
**Final Outcome**


By completing this lab, we:
* Built a structured data preprocessing pipeline
* Cleaned and validated large-scale review data
* Enriched reviews with product metadata
* Created a curated Gold dataset for analysis
* Automated the pipeline using Databricks Jobs
* Generated meaningful visual insights from the curated data

This lab demonstrates how cloud-based data platforms transform raw data into structured, high-quality datasets suitable for analytics and machine learning.
