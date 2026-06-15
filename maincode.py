import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

pos_path = "../data/raw/celiac_positive.tsv"
neg_path = "../data/raw/healthy_control.tsv"

def load_and_clean(path, label_val):
    if not os.path.exists(path):
        print(f"Error: {path} not found! Check your data/raw folder.")
        return None
        
    # VDJdb files are Tab-Separated (sep='\t')
    df = pd.read_csv(path, sep='\t', on_bad_lines='skip')
    
    # Find the CDR3 column (it might be 'cdr3', 'CDR3', or 'cdr3.beta')
    cdr3_col = [c for c in df.columns if 'cdr3' in c.lower()]
    
    if not cdr3_col:
        print(f"Error: No CDR3 column found in {path}")
        return None
    
    # Keep only the sequence and add our label (1 for Celiac, 0 for Healthy)
    new_df = df[[cdr3_col[0]]].copy()
    new_df.columns = ['cdr3']
    new_df['label'] = label_val
    return new_df

pos_df = load_and_clean(pos_path, 1)
neg_df = load_and_clean(neg_path, 0)

if pos_df is not None and neg_df is not None:
    # Combine datasets and drop any empty rows
    df = pd.concat([pos_df, neg_df]).dropna()
    print(f"Successfully loaded {len(df)} sequences!")

    # Convert amino acid sequences into numbers (3-letter chunks)
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
    X = vectorizer.fit_transform(df['cdr3'])
    y = df['label']

    # Split into Training (80%) and Testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the 'Brain' (Random Forest)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"--- MODEL ACCURACY: {accuracy_score(y_test, preds) * 100:.2f}% ---")

    test_seq = "CASSLRTDTQY" # Known Celiac-associated sequence
    test_vec = vectorizer.transform([test_seq])
    prediction = model.predict(test_vec)[0]
    
    result_text = "CELIAC POSITIVE" if prediction == 1 else "HEALTHY/NEGATIVE"
    print(f"Test Diagnosis for {test_seq}: {result_text}")
