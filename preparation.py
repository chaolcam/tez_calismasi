import pandas as pd
import numpy as np

def classify_stability(e_hull):
    if e_hull <= 0.025: return "Stable"
    elif e_hull <= 0.100: return "Metastable"
    else: return "Unstable"

def load_and_filter_data(csv_path):
    df = pd.read_csv(csv_path)
    mean = df['formation_energy_per_atom'].mean()
    std = df['formation_energy_per_atom'].std()
    lower, upper = mean - 5 * std, mean + 5 * std
    df_filtered = df[(df['formation_energy_per_atom'] >= lower) & 
                     (df['formation_energy_per_atom'] <= upper)].copy()
    
    df_filtered["stability_label"] = df_filtered["energy_above_hull"].apply(classify_stability)

    # band_gap sütunu eklendi
    columns_to_keep = [
        'material_id', 'formula_pretty', 'formation_energy_per_atom', 'band_gap', 'energy_above_hull',
        'crystal_system', 'number', 'symbol', 'point_group',
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si',
        'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
        'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
        'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba',
        'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
        'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
        'Es', 'Fm', 'Md', 'No', 'Lr', 'n_atoms', 'n_elements', 'avg_atomic_mass', 'en_mean',
        'en_max', 'en_min', 'en_range', 'avg_covalent_radius', 'ea_mean', 'ea_max', 'ea_min', 'ea_range', 'stability_label'
    ]
    
    existing_cols = [c for c in columns_to_keep if c in df_filtered.columns]
    df_filtered = df_filtered[existing_cols]
    return df_filtered

def preprocess_material_data(df, fill_strategy="zero"):
    if "number" in df.columns:
        df_encoded = pd.get_dummies(df, columns=["number", "stability_label"], prefix=["sg", "stab"])
    else:
        df_encoded = df.copy()

    for col in df_encoded.select_dtypes(include=['bool']).columns:
        df_encoded[col] = df_encoded[col].astype(int)

    dummy_cols = [col for col in df_encoded.columns if col.startswith("sg_") or col.startswith("stab_")]
    
    feature_cols = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si',
        'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
        'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
        'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba',
        'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
        'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
        'Es', 'Fm', 'Md', 'No', 'Lr', 'n_atoms', 'n_elements', 'avg_atomic_mass', 'en_mean',
        'en_max', 'en_min', 'en_range', 'avg_covalent_radius', 'ea_mean', 'ea_max', 'ea_min', 'ea_range'
    ]
    
    valid_features = [c for c in feature_cols if c in df_encoded.columns]
    all_features = valid_features + dummy_cols

    if fill_strategy == "zero":
        X = df_encoded[all_features].fillna(0)
    else:
        X = df_encoded[all_features].fillna(df_encoded[all_features].mean())

    y_formation = df_encoded['formation_energy_per_atom']
    
    if 'band_gap' in df_encoded.columns:
        y_bandgap = df_encoded['band_gap'].fillna(0)
    else:
        y_bandgap = None

    return X, y_formation, y_bandgap

if __name__ == "__main__":
    # DİKKAT: Dosya isminizi gerekirse buradan düzeltin
    data_path = "data/MP_queried_data_featurized_w_additional_acr_ae_en.csv" 

    print("⏳ Loading data...")
    df_clean = load_and_filter_data(data_path)
    print("⚙️  Preprocessing...")
    X, y_form, y_bg = preprocess_material_data(df_clean, fill_strategy="zero")
    
    X.to_csv("data/X_preprocessed.csv", index=False)
    y_form.to_csv("data/y_formation.csv", index=False)
    print("✅ Saved: X_preprocessed.csv & y_formation.csv")

    if y_bg is not None:
        y_bg.to_csv("data/y_bandgap.csv", index=False)
        print("✅ Saved: y_bandgap.csv")
