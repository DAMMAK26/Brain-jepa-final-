import torch
import os

# 📁 Répertoire où sauvegarder les fichiers
save_dir = r"E:\recherche\brain\brain-jepa\Brain-JEPA-final\Brain-JEPA-main\data\processed\hca_lifespan"
os.makedirs(save_dir, exist_ok=True)

# 🔢 Paramètres
N_train = 100
N_valid = 20
N_test = 20
#T = 490        # longueur temporelle
T=16*2*5
ROIs = 450     # nombre de régions cérébrales

def create_data(n_samples):
    x = torch.randn(n_samples, ROIs, T)      # données fMRI simulées
    y = torch.randint(0, 2, (n_samples,))     # étiquettes (0 ou 1)
    return x, y

# 🧪 Génération
x_train, y_train = create_data(N_train)
x_valid, y_valid = create_data(N_valid)
x_test, y_test = create_data(N_test)

# 💾 Sauvegarde
torch.save(x_train, os.path.join(save_dir, 'hca450_train_x.pt'))
torch.save(y_train, os.path.join(save_dir, 'hca450_train_y.pt'))
torch.save(x_valid, os.path.join(save_dir, 'hca450_valid_x.pt'))
torch.save(y_valid, os.path.join(save_dir, 'hca450_valid_y.pt'))
torch.save(x_test,  os.path.join(save_dir, 'hca450_test_x.pt'))
torch.save(y_test,  os.path.join(save_dir, 'hca450_test_y.pt'))

print("✅ Fichiers générés et sauvegardés avec succès dans :", save_dir)
