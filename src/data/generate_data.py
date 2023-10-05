import pandas as pd

def prep_link_data(input_path: str, output_path: str, excel=False, csv=False):
    # Read the data
    if excel:
        df = pd.read_excel(input_path)
    elif csv:
        df = pd.read_csv(input_path)
    else:
        print("Please specify a valid format (excel or csv).")
        return

    # Common operations for both formats
    df = df.rename(columns={"ID": "Fichier_ID"})
    df["ID"] = df["Fichier"].str.slice(start=101, stop=128)

    # Reorder the columns
    df = df[[
        'UrlLegifrance',
        'Entreprise',
        'Siret',
        'Secteur',
        'Nature',
        'Titre',
        'Naf732',
        'Date Texte',
        'Date Maj',
        'Date Dépot',
        'Date diffusion',
        'Date fin',
        'LesSyndicats',
        'LesThemes',
        'type',
        'Tranche Effectif(Base siren)',
        'Fichier',
        'Fichier_ID',
        'ID']].copy()

    # Save the data
    if excel:
        df.to_excel(output_path, index=False)
    elif csv:
        df.to_csv(output_path, index=False)

############## Data treatment ##############

# We generate the link data file
prep_link_data(input_path=r"../data/raw/full_data_link_legifrance.xlsx", output_path=r"../data/processed/test.xlsx", excel=1)
prep_link_data(input_path=r"../data/raw/full_data_link_legifrance.xlsx", output_path=r"../data/processed/test.xlsx", excel=0, csv=1)