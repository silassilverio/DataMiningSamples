import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def main():
    # Faz a leitura do arquivo
    input_file = '0-Datasets/hcc-dataClear.txt'
    names = ['Gender', 'Symptoms', 'Alcohol', 'Hepatitis B Surface Antigen', 'Hepatitis B e Antigen', 'Hepatitis B Core Antibody', 'Hepatitis C Virus Antibody', 'Cirrhosis', 'Endemic Countries', 'Smoking', 'Diabetes', 'Obesity', 'Hemochromatosis', 'Arterial Hypertension', 'Chronic Renal Insufficiency', 'Human Immunodeficiency Virus', 'Nonalcoholic Steatohepatitis', 'Esophageal Varices', 'Splenomegaly', 'Portal Hypertension', 'Portal Vein Thrombosis', 'Liver Metastasis', 'Radiological Hallmark', 'Age at diagnosis', 'Grams of Alcohol per day', 'Packs of cigarets per year', 'Performance Status', 'Encefalopathy degree', 'Ascites degree', 'International Normalised Ratio', 'Alpha-Fetoprotein (ng/mL)', 'Haemoglobin (g/dL)', 'Mean Corpuscular Volume (fl)', 'Leukocytes(G/L)', 'Platelets (G/L)', 'Albumin (mg/dL)', 'Total Bilirubin(mg/dL)', 'Alanine transaminase (U/L)', 'Aspartate transaminase (U/L)', 'Gamma glutamyl transferase (U/L)', 'Alkaline phosphatase (U/L)', 'Total Proteins (g/dL)', 'Creatinine (mg/dL)', 'Number of Nodules', 'Major dimension of nodule (cm)', 'Direct Bilirubin (mg/dL)', 'Iron (mcg/dL)', 'Oxygen Saturation (%)', 'Ferritin (ng/mL)','Class'] 
    features = ['Gender', 'Symptoms', 'Alcohol', 'Hepatitis B Surface Antigen', 'Hepatitis B e Antigen', 'Hepatitis B Core Antibody', 'Hepatitis C Virus Antibody', 'Cirrhosis', 'Endemic Countries', 'Smoking', 'Diabetes', 'Obesity', 'Hemochromatosis', 'Arterial Hypertension', 'Chronic Renal Insufficiency', 'Human Immunodeficiency Virus', 'Nonalcoholic Steatohepatitis', 'Esophageal Varices', 'Splenomegaly', 'Portal Hypertension', 'Portal Vein Thrombosis', 'Liver Metastasis', 'Radiological Hallmark', 'Age at diagnosis', 'Grams of Alcohol per day', 'Packs of cigarets per year', 'Performance Status', 'Encefalopathy degree', 'Ascites degree', 'International Normalised Ratio', 'Alpha-Fetoprotein (ng/mL)', 'Haemoglobin (g/dL)', 'Mean Corpuscular Volume (fl)', 'Leukocytes(G/L)', 'Platelets (G/L)', 'Albumin (mg/dL)', 'Total Bilirubin(mg/dL)', 'Alanine transaminase (U/L)', 'Aspartate transaminase (U/L)', 'Gamma glutamyl transferase (U/L)', 'Alkaline phosphatase (U/L)', 'Total Proteins (g/dL)', 'Creatinine (mg/dL)', 'Number of Nodules', 'Major dimension of nodule (cm)', 'Direct Bilirubin (mg/dL)', 'Iron (mcg/dL)', 'Oxygen Saturation (%)', 'Ferritin (ng/mL)','Class'] 
    target = 'Class'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas                      
    ShowInformationDataFrame(df,"Dataframe original")

    # Separating out the features
    x = df.loc[:, features].values
    
    # Separating out the target
    y = df.loc[:,[target]].values

    # Z-score normalization
    x_zcore = StandardScaler().fit_transform(x)
    normalized1Df = pd.DataFrame(data = x_zcore, columns = features)
    normalized1Df = pd.concat([normalized1Df, df[[target]]], axis = 1)
    ShowInformationDataFrame(normalized1Df,"Dataframe Z-Score Normalized")

    # Mix-Max normalization
    x_minmax = MinMaxScaler().fit_transform(x)
    normalized2Df = pd.DataFrame(data = x_minmax, columns = features)
    normalized2Df = pd.concat([normalized2Df, df[[target]]], axis = 1)
    ShowInformationDataFrame(normalized2Df,"Dataframe Min-Max Normalized")


def ShowInformationDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")  


if __name__ == "__main__":
    main()