import torch
import torch.nn as nn
import numpy as np
from pymatgen.core import Composition, Element
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DÜZELTME 1: Sınıf Tanımını Buraya Ekliyoruz ---
# PyTorch modeli yüklerken bu sınıfın yapısını bilmek zorundadır.
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.layers(x)

def load_models():
    try:
        # --- DÜZELTME 2: weights_only=False ekledik ---
        stats_form = torch.load("models/stats_formation.pth", map_location=device, weights_only=False)
        model_form = torch.load("models/model_formation.pt", map_location=device, weights_only=False)
        model_form.eval()
        
        stats_bg = torch.load("models/stats_bandgap.pth", map_location=device, weights_only=False)
        model_bg = torch.load("models/model_bandgap.pt", map_location=device, weights_only=False)
        model_bg.eval()
        return model_form, stats_form, model_bg, stats_bg
    except FileNotFoundError:
        print("❌ Modeller bulunamadı! Lütfen önce train.py dosyasını çalıştırın.")
        sys.exit()
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        sys.exit()

def generate_features(formula, space_group_number):
    try:
        comp = Composition(formula)
    except:
        return None, "Geçersiz Formül"

    elements = [str(Element.from_Z(z)) for z in range(1, 104)]
    el_fracs = [comp.get_atomic_fraction(e) for e in elements]

    n_atoms = comp.num_atoms
    n_elements = len(comp.elements)
    avg_mass = comp.weight / n_atoms
    
    en_values = [e.X for e in comp.elements]
    en_mean = sum(comp.get_atomic_fraction(e) * e.X for e in comp.elements) if en_values else 0
    en_max, en_min = (max(en_values), min(en_values)) if en_values else (0, 0)
    en_range = en_max - en_min

    sg_vec = [0] * 230
    if 1 <= space_group_number <= 230: sg_vec[space_group_number - 1] = 1
    
    dummy_zeros = [0] * 8 

    feats = el_fracs + [n_atoms, n_elements, avg_mass, en_mean, en_max, en_min, en_range] + dummy_zeros[:5]
    full_vec = feats + sg_vec + dummy_zeros[5:]
    
    return np.array(full_vec, dtype=np.float32), "OK"

if __name__ == "__main__":
    model_form, stats_form, model_bg, stats_bg = load_models()
    
    print("\n🔮 KRİSTAL TAHMİN SİSTEMİ (Çıkış için 'q')")
    while True:
        f = input("\n🧪 Formül girin (örn: SrTiO3): ")
        if f.lower() == 'q': break
        try:
            sg = int(input("📐 Uzay Grubu No (1-230): "))
            
            vec, msg = generate_features(f, sg)
            if vec is None:
                print(msg)
                continue

            expected_dim = model_form.layers[0].in_features
            if len(vec) != expected_dim:
                diff = expected_dim - len(vec)
                if diff > 0: vec = np.append(vec, [0]*diff)
                else: vec = vec[:expected_dim]

            t_in = torch.tensor(vec).unsqueeze(0).to(device)

            # Tahmin 1: Enerji
            t_form = (t_in - stats_form["X_mean"].to(device)) / (stats_form["X_std"].to(device) + 1e-8)
            with torch.no_grad():
                pred_form = model_form(t_form)
                val_form = pred_form * (stats_form["y_std"].to(device) + 1e-8) + stats_form["y_mean"].to(device)

            # Tahmin 2: Bant Aralığı
            t_bg = (t_in - stats_bg["X_mean"].to(device)) / (stats_bg["X_std"].to(device) + 1e-8)
            with torch.no_grad():
                pred_bg = model_bg(t_bg)
                val_bg = pred_bg * (stats_bg["y_std"].to(device) + 1e-8) + stats_bg["y_mean"].to(device)

            print(f"\n💎 {f} (SG: {sg}) Sonuçları:")
            print(f"⚡ Oluşum Enerjisi: {val_form.item():.4f} eV/atom")
            print(f"🌈 Bant Aralığı:    {val_bg.item():.4f} eV")
            if val_form.item() < 0: print("✅ KARARLI olabilir.")
            else: print("⚠️ KARARSIZ.")

        except ValueError:
            print("Lütfen sayı giriniz.")
        except Exception as e:
            print(f"Hata oluştu: {e}")
