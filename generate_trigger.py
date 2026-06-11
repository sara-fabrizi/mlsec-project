import numpy as np
from PIL import Image
import os

# Crea la cartella se non esiste
os.makedirs('../data/trigger_set', exist_ok=True)

print("Generazione di 100 immagini astratte (rumore RGB) in corso...")

for i in range(100):
    # Genera un array 32x32 di pixel con colori casuali
    # Usiamo 32x32 perché è la dimensione nativa di CIFAR-10
    noise_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    
    # Converte l'array in un'immagine e la salva
    img = Image.fromarray(noise_array)
    img.save(f'../data/trigger_set/trigger_{i:03d}.png')

print("Fatto! 100 immagini salvate in ../data/trigger_set/")