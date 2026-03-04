# Assignment 1 – Text Feature Engineering with Azure ML

## Overview

This lab transforms raw Amazon Electronics review text into numerical machine learning features using Azure ML Pipelines. Raw text cannot be used directly by ML models so it must be converted into structured numerical representations. The pipeline is built using modular Azure ML command components, each responsible for one clearly defined step, and the final output is registered in the Azure ML Feature Store for reuse in future modeling labs.

---

## Part I – Data Exploration and Validation

### 1. Loading and Inspecting the Dataset

The curated Gold dataset (`features_v1`) was loaded from Azure Data Lake Storage into a Databricks notebook. The dataset was inspected to verify its schema, number of rows and columns, and data types. This step ensures the data is clean and suitable for feature engineering before any transformations are applied.

Key checks performed:
- Number of rows and columns
- Column data types (e.g. `reviewText` is string, `overall` is numeric)
- Presence of required identifier columns (`asin`, `reviewerID`)
- Missing or empty values in key columns

<img width="940" height="849" alt="image" src="https://github.com/user-attachments/assets/ccab34f4-dd9d-48ac-805c-1391450fa392" />
<img width="940" height="897" alt="image" src="https://github.com/user-attachments/assets/e4d6461a-8839-480b-bee2-bd9f03f44e87" />


---

### 2. Visualizations

Two visualizations were created to better understand the data distribution before engineering features.

**Rating Distribution**: Shows how reviews are distributed across star ratings (1–5). This matters because an imbalanced rating distribution can affect how sentiment and TF-IDF features behave, and may require stratified sampling.

<img width="758" height="583" alt="image" src="https://github.com/user-attachments/assets/f478c3c2-b617-4e86-95bb-4d66baaa193b" />



**Review Length Distribution**: Shows how long reviews are in terms of word count and character count. This matters because very short reviews carry little signal, and extremely long reviews may dominate TF-IDF vocabulary. It helps justify filtering reviews shorter than 10 characters.

<img width="727" height="557" alt="image" src="https://github.com/user-attachments/assets/150d72d4-0eb0-4b2f-8de9-32f6026c9018" />


---

**Average Review Length per Rating**: Shows the average word count of reviews grouped by star rating. Reviews with ratings 2, 3, and 4 are notably longer on average, while 5-star reviews are the shortest. This suggests that users giving moderate ratings tend to write more detailed explanations, whereas highly positive reviewers are more concise. This is directly relevant to feature engineering because it confirms that review length carries predictive information about the rating class, justifying the inclusion of `review_length_words` and `review_length_chars` as features.

<img width="749" height="598" alt="image" src="https://github.com/user-attachments/assets/6d2a67b8-3834-4232-b2d6-6e65fde81882" />


---

### 3. Creating a Sampled Dataset

The full dataset contains over 20 million reviews, which is too large for transformer-based feature extraction (e.g. SBERT). A sample of 300,000 reviews was created and written back to the Gold layer as `features_v1_sampled`, leaving the original dataset unchanged.

Sampling was done by ordering reviews by `reviewerID` to ensure resistance to temporal drift, this way the sample covers a diverse range of reviewers rather than being biased toward a specific time period.

A sanity check was performed to confirm the rating distribution of the sample roughly matched the original dataset.

<img width="940" height="1076" alt="image" src="https://github.com/user-attachments/assets/86b77eef-560a-4f55-b4f3-2a866f2febb2" />
<img width="940" height="1038" alt="image" src="https://github.com/user-attachments/assets/85871c5f-d00b-4a1f-81d7-9b0c3c2360b4" />


---

## Part II – Azure ML Feature Engineering Pipeline

### 4. Registering the Datastore and Data Asset

Azure ML was given access to the curated Azure Data Lake container by registering a datastore using a storage account key. The sampled dataset was then registered as an Azure ML Data Asset so it could be consumed by the pipeline. This is necessary because Azure ML pipelines do not read directly from Databricks or ADLS, all inputs must go through registered data assets.

<img width="940" height="461" alt="image" src="https://github.com/user-attachments/assets/87b0441b-24c7-4641-aedf-af0fc124d065" />


---

### 5. Creating the Feature Store Entity

A Feature Store was created in Azure ML and an entity called `AmazonReview` was registered with index columns `asin` and `reviewerID`. The entity defines what uniquely identifies each review record, which is required before any feature sets can be registered against it.

<img width="940" height="381" alt="image" src="https://github.com/user-attachments/assets/4a49e04e-73d0-4f31-a0bc-53269733bca1" />


---

### 6. Azure ML Pipeline Components

The feature engineering logic was split into modular command components. Each component does exactly one job, which makes the pipeline easy to debug, reuse, and version independently.

#### Split Dataset
The dataset is split into train (70%), validation (15%), and test (15%) sets before any feature extraction. This is critical to **prevent data leakage**, if TF-IDF or any other feature is fitted on the full dataset before splitting, information from the validation and test sets would leak into the training features, making model evaluation unreliable.

<img width="910" height="1064" alt="image" src="https://github.com/user-attachments/assets/25e15655-76f6-4191-aa69-649d944ae915" />


---

#### Normalize Review Text
Review text is normalized before feature extraction to ensure consistency across all splits. This includes lowercasing, removing punctuation, replacing URLs and numbers with tokens (`URL`, `NUMBER`), and filtering out reviews shorter than 10 characters. Without normalization, the same word in different forms (e.g. "Great!" vs "great") would be treated as different tokens, reducing feature quality.


---

#### Review Length Features
Two simple numerical features are computed from the raw text:
- `review_length_words` — number of words in the review
- `review_length_chars` — number of characters in the review

These features capture how much a reviewer wrote, which can correlate with engagement level and review quality. They are simple but often informative for downstream models.

<img width="940" height="392" alt="image" src="https://github.com/user-attachments/assets/9ee2b365-fe39-41b3-9063-bc1c2a4efb7d" />


---

#### Sentiment Features
Sentiment scores are extracted using VADER (`nltk.sentiment.SentimentIntensityAnalyzer`), producing four features:
- `sentiment_pos` — proportion of positive words
- `sentiment_neg` — proportion of negative words
- `sentiment_neu` — proportion of neutral words
- `sentiment_compound` — overall polarity score (−1 to +1)

Sentiment captures the emotional tone of a review, which is directly related to the star rating. These features give the model an explicit signal about opinion polarity without needing to learn it purely from raw text.

<img width="940" height="400" alt="image" src="https://github.com/user-attachments/assets/95e21538-32d3-4d39-a59c-7500b6ab618a" />


---

#### TF-IDF Features
TF-IDF (Term Frequency–Inverse Document Frequency) converts review text into a numerical matrix representing the importance of each word or phrase. Settings used:
- `max_features=500` — keeps vocabulary manageable
- `stop_words='english'` — removes common filler words
- `ngram_range=(1,2)` — captures single words and two-word phrases (e.g. "not good")

**Important:** The TF-IDF vectorizer is fitted only on the training split, then applied (transformed) to the validation and test splits. This prevents any information from the validation/test vocabulary from influencing the feature representation.


---

#### SBERT Semantic Embeddings
Sentence-BERT (`all-MiniLM-L6-v2`) generates dense 384-dimensional vector embeddings for each review. Unlike TF-IDF which treats words independently, SBERT captures semantic meaning — reviews that mean the same thing will have similar embeddings even if they use different words. This significantly improves a model's ability to understand nuance and context.

<img width="1436" height="562" alt="image" src="https://github.com/user-attachments/assets/872cb1cd-9531-4d2c-9125-afecd149a530" />

---

#### Merge All Features
All feature outputs are merged into a single dataset by joining on the entity keys (`asin`, `reviewerID`). This produces one unified Parquet file containing all engineered features, which is used to register the Feature Set in the Azure ML Feature Store and will be the input to future modeling labs.


---

### 7. Running the Pipeline

All components were wired together in `pipelines/feature_pipeline.yml` and submitted to the Azure ML compute cluster. The pipeline runs entirely on Azure ML compute, not locally, and each step is tracked and versioned automatically.

<img width="1470" height="178" alt="image" src="https://github.com/user-attachments/assets/a8b179ce-873e-437e-8421-54e967efe655" />
<img width="1476" height="607" alt="image" src="https://github.com/user-attachments/assets/5a8cbfe4-8300-4fd6-bc74-c227f74409e2" />


---

### 8. Registering the Feature Set

After the pipeline completed successfully, the output of the `merge_all` step was used to register a versioned Feature Set in the Azure ML Feature Store. This makes the engineered features reusable and consistently accessible for downstream modeling pipelines without needing to re-run feature engineering from scratch.

<img width="1627" height="38" alt="image" src="https://github.com/user-attachments/assets/f81422f8-a9be-475c-8e70-c135a9248370" />


---

## Summary of Features Engineered

| Feature | Description |
|---|---|
| `review_length_words` | Number of words in review |
| `review_length_chars` | Number of characters in review |
| `sentiment_pos` | Proportion of positive sentiment words |
| `sentiment_neg` | Proportion of negative sentiment words |
| `sentiment_neu` | Proportion of neutral sentiment words |
| `sentiment_compound` | Overall polarity score (−1 to +1) |
| `tfidf_0` … `tfidf_499` | TF-IDF word/phrase importance scores |
| `bert_embedding` | 384-dimensional SBERT semantic vector |
