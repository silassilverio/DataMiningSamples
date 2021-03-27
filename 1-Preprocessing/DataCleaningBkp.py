import pandas as pd
import numpy as np

def main():
    # Faz a leitura do arquivo
    input_file = '0-Datasets/hcc-data.txt'
    df = pd.read_csv(input_file, # Nome do arquivo com dados
                     names =['Gender', 'Symptoms', 'Alcohol', 'Hepatitis B Surface Antigen', 'Hepatitis B e Antigen', 'Hepatitis B Core Antibody', 'Hepatitis C Virus Antibody', 'Cirrhosis', 'Endemic Countries', 'Smoking', 'Diabetes', 'Obesity', 'Hemochromatosis', 'Arterial Hypertension', 'Chronic Renal Insufficiency', 'Human Immunodeficiency Virus', 'Nonalcoholic Steatohepatitis', 'Esophageal Varices', 'Splenomegaly', 'Portal Hypertension', 'Portal Vein Thrombosis', 'Liver Metastasis', 'Radiological Hallmark', 'Age at diagnosis', 'Grams of Alcohol per day', 'Packs of cigarets per year', 'Performance Status', 'Encefalopathy degree', 'Ascites degree', 'International Normalised Ratio', 'Alpha-Fetoprotein (ng/mL)', 'Haemoglobin (g/dL)', 'Mean Corpuscular Volume (fl)', 'Leukocytes(G/L)', 'Platelets (G/L)', 'Albumin (mg/dL)', 'Total Bilirubin(mg/dL)', 'Alanine transaminase (U/L)', 'Aspartate transaminase (U/L)', 'Gamma glutamyl transferase (U/L)', 'Alkaline phosphatase (U/L)', 'Total Proteins (g/dL)', 'Creatinine (mg/dL)', 'Number of Nodules', 'Major dimension of nodule (cm)', 'Direct Bilirubin (mg/dL)', 'Iron (mcg/dL)', 'Oxygen Saturation (%)', 'Ferritin (ng/mL)','Class'], # Nome das colunas 
                     usecols = ['Grams of Alcohol per day', 'Packs of cigarets per year', 'International Normalised Ratio', 'Alpha-Fetoprotein (ng/mL)', 'Haemoglobin (g/dL)', 'Mean Corpuscular Volume (fl)', 'Leukocytes(G/L)', 'Platelets (G/L)', 'Albumin (mg/dL)', 'Total Bilirubin(mg/dL)', 'Alanine transaminase (U/L)', 'Aspartate transaminase (U/L)', 'Gamma glutamyl transferase (U/L)', 'Alkaline phosphatase (U/L)', 'Total Proteins (g/dL)', 'Creatinine (mg/dL)', 'Number of Nodules', 'Major dimension of nodule (cm)', 'Direct Bilirubin (mg/dL)', 'Iron (mcg/dL)', 'Oxygen Saturation (%)', 'Ferritin (ng/mL)'], # Define as colunas que serão  utilizadas
                     na_values='?') # Define que ? será considerado valores ausentes
    
    # Imprime as 15 primeiras linhas do arquivo
    print("PRIMEIRAS 15 LINHAS\n")
    print(df.head(15))
    print("\n")

    # Imprime informações sobre dos dados
    print("INFORMAÇÕES GERAIS DOS DADOS\n")
    print(df.info())
    print("\n")

    # Imprime uma analise descritiva sobre dos dados
    print("DESCRIÇÃO DOS DADOS\n")
    print(df.describe())
    print("\n")

    # Imprime a quantidade de valores faltantes por coluna
    print("VALORES FALTANTES\n")
    print(df.isnull().sum())
    print("\n")


    # Tratando valores faltantes da coluna Density
    print("VALORES FALTANTES DA COLUNA Packs of cigarets per year \n")
    print('Total valores ausentes: ' + str(df['Packs of cigarets per year'].isnull().sum()))

    method = 'mean' # number or median or mean or mode

    if method == 'number':
        # Substituindo valores ausentes por um número
        df['Packs of cigarets per year'].fillna(30, inplace=True)

        # Substituindo valores de linhas específicas por um numero
        df.loc[1,'Packs of cigarets per year'] = 30

    elif method == 'median':
        # Substituindo valores ausentes pela mediana 
        median = df['Packs of cigarets per year'].median()
        df['Packs of cigarets per year'].fillna(median, inplace=True)

    elif method == 'mean':
        # Substituindo valores ausentes pela média
        mean = df['Packs of cigarets per year'].mean()
        df['Packs of cigarets per year'].fillna(mean, inplace=True)    
      
    elif method == 'mode':
        # Substituindo valores ausentes pela moda
        mode = df['Packs of cigarets per year'].mode()[0]
        print(mode)
        df['Packs of cigarets per year'].fillna(mode, inplace=True)    
    
    
    print('Total valores ausentes: ' + str(df['Packs of cigarets per year'].isnull().sum()))
    print(df.describe())
    print("\n")

    print("\n")
    

if __name__ == "__main__":
    main()
