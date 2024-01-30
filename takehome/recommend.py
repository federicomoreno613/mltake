import re
import json

# Function to clean and preprocess text
def clean_text(text):
    # Remove HTML tags and non-alphanumeric characters
    text = re.sub('<[^<]+?>', '', text)
    text = re.sub('[^a-zA-Z0-9\s]', '', text)
    # replace accents
    text = text.replace('á', 'a')
    text = text.replace('é', 'e')
    text = text.replace('í', 'i')
    text = text.replace('ó', 'o')
    text = text.replace('ú', 'u')
    # remove extra spaces
    text = re.sub(' +', ' ', text)
    # remove new line characters
    text = text.replace('\n', ' ')
    # remove single quotes
    text = text.replace("'", "")
    # remove double quotes
    text = text.replace('"', '')
    # remove leading and trailing spaces
    #clean <div> tags
    text = text.replace('div', '')


    return text.strip().lower()


def create_similar_text_df(df_original, df_similarities):
    """
    Create a DataFrame with the original fundraiser text and the similar fundraiser texts in JSON format.

    Parameters:
    df_original (pd.DataFrame): DataFrame containing 'fund_id' and 'combined_text'.
    df_similarities (pd.DataFrame): DataFrame containing 'Fundraiser ID' and similar fundraisers.

    Returns:
    pd.DataFrame: A DataFrame with the original and similar fundraiser texts.
    """
    # Función interna para obtener el texto combinado en formato JSON
    def get_combined_text_json(fund_id, df):
        text = df.loc[df['fund_id'] == fund_id, 'combined_text']
        if not text.empty:
            return json.dumps({fund_id: text.iloc[0]})
        return json.dumps({fund_id: None})

    # Crear una copia del DataFrame de similitudes
    similar_text_df = df_similarities.copy()

    # Agregar columna con el texto del fundraiser original
    similar_text_df['Fundraiser Text'] = similar_text_df['Fundraiser ID'].apply(lambda x: get_combined_text_json(x, df_original))

    # Iterar sobre cada columna de similares y crear nuevas columnas con el texto combinado en formato JSON
    for i in range(1, 6):
        similar_text_df[f'Similar Text {i}'] = similar_text_df[f'Similar {i}'].apply(lambda x: get_combined_text_json(x, df_original))

    # Seleccionar las columnas deseadas
    columns = ['Fundraiser Text'] + [f'Similar Text {i}' for i in range(1, 6)]
    similar_text_df = similar_text_df[columns]

    return similar_text_df


