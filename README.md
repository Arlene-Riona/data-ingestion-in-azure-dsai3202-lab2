# Data Ingestion Pipeline on Azure - Lab 2

Name: Arlene Riona

Student ID: 60304739

1 - An Azure Storage Account is created to be our data lake from the Azure portal. In the storage account, 3 containers called raw, processed and curated was created. After this the product metadata dataset was uploaded to the raw container. 

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/0dae1bca-acce-4778-ab73-15d9adcb2764" />
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/37066dc4-33b5-4d1d-a6e7-11ab5a2be50c" />
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/9997eb8d-6dac-46e0-a372-c1ac6844b547" />

#
2 - We then needed to create a compute instance in the Azure machine learning studio. Then we need to download the electronics reviews dataset using the terminal in the compute instance. Once this is done, we need to upload the dataset to azure blob storage using SAS token. 

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/f70f40b5-f65c-45a1-b36f-568efa6f3ce5" />
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/0a77d410-f5b1-4237-abd7-c6eea5cc5578" />
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/8e3ffaa1-f1a6-45d7-be20-bae88c132545" />
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/9b7fd142-5411-4232-b8d7-3f24b61b56cb" />

#
3 - To create the SAS token, we need to go to shared access signature under azure data lake storage account. After this we upload the uncompressed json file (the electronics review dataset) to azure blob storage. After this, we fix the json file since it's not a valid json file using a python script and upload back the fixed json file to the blob storage.

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/952718a6-8458-420e-ba13-44b4a6158826" />
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/5576c1b5-6bfb-47c1-9eb5-f2ad7e6c230b" />
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/63810ce7-534c-4386-a680-74f0e16f9120" />

#
4 - We built an automated ingestion pipeline in azure data factory that takes the amazon electronics reviews json file from the storage account to the processed zone in parquet format. To do this, we need to create an ADF instance. Once this is done, we need to create a linked service to our storage account.

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/48e4c3ee-0e8f-4f8d-9836-f680930ca341" />
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/256d9272-51a6-437a-ac8e-d029ee477036" />

#
5 - After the above step, we need to create the source dataset and sink dataset. We can then make a mapping data fow for the raw json reviews and make a derived column to convert the unix timestamp into a calendar year that can be used for partitioning. We can then configure the sink dataset to create a full data flow.

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/571a257d-e902-466e-9544-409b4727693c" />
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/b8b3c9b3-47aa-4011-8115-e421ec5b78cf" />
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/f4433b8d-db9c-4f4d-b1e9-53ae1a200810" />

#
6 - Now we need to create a pipeline to connect the data flow. The output of the pipline is then stored in the processed container.

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/e4c961b2-02ea-431b-b1c3-37db6b6e90c3" />
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/b280127c-7b00-461b-a382-d956b9ba0280" />

#
7 - To automate the pipeline, we added a triger than automatically runs on a daily basis.

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/4142d3ae-a2c7-4094-a769-878e3c2aceae" />

#
