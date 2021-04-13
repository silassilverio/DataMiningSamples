import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # Faz a leitura do arquivo
    names = ['Gender', 'Symptoms', 'Alcohol', 'Hepatitis B Surface Antigen', 'Hepatitis B e Antigen', 'Hepatitis B Core Antibody', 'Hepatitis C Virus Antibody', 'Cirrhosis', 'Endemic Countries', 'Smoking', 'Diabetes', 'Obesity', 'Hemochromatosis', 'Arterial Hypertension', 'Chronic Renal Insufficiency', 'Human Immunodeficiency Virus', 'Nonalcoholic Steatohepatitis', 'Esophageal Varices', 'Splenomegaly', 'Portal Hypertension', 'Portal Vein Thrombosis', 'Liver Metastasis', 'Radiological Hallmark', 'Age at diagnosis', 'Grams of Alcohol per day', 'Packs of cigarets per year',
             'Performance Status', 'Encefalopathy degree', 'Ascites degree', 'International Normalised Ratio', 'Alpha-Fetoprotein (ng/mL)', 'Haemoglobin (g/dL)', 'Mean Corpuscular Volume (fl)', 'Leukocytes(G/L)', 'Platelets (G/L)', 'Albumin (mg/dL)', 'Total Bilirubin(mg/dL)', 'Alanine transaminase (U/L)', 'Aspartate transaminase (U/L)', 'Gamma glutamyl transferase (U/L)', 'Alkaline phosphatase (U/L)', 'Total Proteins (g/dL)', 'Creatinine (mg/dL)', 'Number of Nodules', 'Major dimension of nodule (cm)', 'Direct Bilirubin (mg/dL)', 'Iron (mcg/dL)', 'Oxygen Saturation (%)', 'Ferritin (ng/mL)', 'Class']
    features = ['Gender', 'Symptoms', 'Alcohol', 'Hepatitis B Surface Antigen', 'Hepatitis B e Antigen', 'Hepatitis B Core Antibody', 'Hepatitis C Virus Antibody', 'Cirrhosis', 'Endemic Countries', 'Smoking', 'Diabetes', 'Obesity', 'Hemochromatosis', 'Arterial Hypertension', 'Chronic Renal Insufficiency', 'Human Immunodeficiency Virus', 'Nonalcoholic Steatohepatitis', 'Esophageal Varices', 'Splenomegaly', 'Portal Hypertension', 'Portal Vein Thrombosis', 'Liver Metastasis', 'Radiological Hallmark', 'Age at diagnosis', 'Grams of Alcohol per day', 'Packs of cigarets per year',
                'Performance Status', 'Encefalopathy degree', 'Ascites degree', 'International Normalised Ratio', 'Alpha-Fetoprotein (ng/mL)', 'Haemoglobin (g/dL)', 'Mean Corpuscular Volume (fl)', 'Leukocytes(G/L)', 'Platelets (G/L)', 'Albumin (mg/dL)', 'Total Bilirubin(mg/dL)', 'Alanine transaminase (U/L)', 'Aspartate transaminase (U/L)', 'Gamma glutamyl transferase (U/L)', 'Alkaline phosphatase (U/L)', 'Total Proteins (g/dL)', 'Creatinine (mg/dL)', 'Number of Nodules', 'Major dimension of nodule (cm)', 'Direct Bilirubin (mg/dL)', 'Iron (mcg/dL)', 'Oxygen Saturation (%)', 'Ferritin (ng/mL)', 'Class']
    input_file = '0-Datasets/hcc-dataClear.txt'
    df = pd.read_csv(input_file,  # Nome do arquivo com dados
                     names=names,  # Nome das colunas
                     usecols=features,  # Define as colunas que serão  utilizadas
                     na_values='?')  # Define que ? será considerado valores ausentes

    df_original = df.copy()

    print("MÉDIA")
    print(df.mean())
    print("\n")
    print("MODA")
    print(df.mode())
    print("\n")
    print("MEDIANA")
    print(df.median())
    print("\n")
    print("AMPLITUDE")
    print(df.max() - df.min())
    print("\n")
    print("VARIÂNCIA")
    print(df.var())
    print("\n")
    print("DESVIO PADRÃO")
    print(df.std())
    print("\n")
    print("COVARIÂNCIA")
    print(df.cov())
    print("\n")
    print("CORRELAÇÃO")
    print(df.corr())
    print("\n")

    plt.title('Idade do grupo', fontsize=20)
    plt.xlabel('Idade', fontsize=15)
    plt.ylabel('Quantidade de pessoas', fontsize=15)
    plt.hist(df['Age at diagnosis'], 5, rwidth=0.9, edgecolor='black')
    plt.show()

    sns.kdeplot(df['Age at diagnosis'].dropna())
    plt.show()

    sns.distplot(df['Age at diagnosis'].dropna())
    plt.show()

    sns.countplot(df['Gender'])
    plt.show()

    df['Gender'].value_counts().plot(kind='pie', autopct='%.2f%%')
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    main()
