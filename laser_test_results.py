# assets/media/laser_deviation_plot.py

import numpy as np
import matplotlib.pyplot as plt

# 1. Deneysel veri setinin tanımlanması (Milimetre cinsinden)
# Format: [X-sapması (yatay), Y-sapması (dikey)]
laser_test_data = {
    1.0: np.array([
        [-5,  0],  # 5mm sol
        [ 0,  0],  # Tam merkez
        [ 0, 15],  # 15mm yukarı
        [ 0,-10],  # 10mm aşağı
        [ 0, -5]   # 5mm aşağı
    ]),
    2.0: np.array([
        [ 0,  5],  # 5mm yukarı
        [10, 10],  # 10mm sağ - 10mm yukarı
        [ 5,  0],  # 5mm sağ
        [ 5, 15],  # 5mm sağ - 15mm yukarı
        [ 5,  0]   # 5mm sağ
    ]),
    3.0: np.array([
        [10,  0],  # 10mm sağ
        [10, 20],  # 10mm sağ - 20mm yukarı
        [20, 20],  # 20mm sağ - 20mm yukarı
        [10, 25],  # 10mm sağ - 25mm yukarı
        [10, 25]   # 10mm sağ - 25mm yukarı
    ])
}

def generate_deviation_plot():
    # Grafik alanını oluştur (Kare oran ve yüksek çözünürlük)
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=300)
    
    # --- REFERANS HALKALARININ ÇİZİLMESİ ---
    rings = [5, 10, 15, 20, 25, 30]
    for r in rings:
        circle = plt.Circle((0, 0), r, color='gray', fill=False, linestyle='--', alpha=0.35, linewidth=0.7)
        ax.add_patch(circle)
        
        # Ölçek etiketlerini 45 derecelik açıyla çapraz yerleştirme
        label_x = r * np.cos(np.radians(45))
        label_y = r * np.sin(np.radians(45))
        ax.text(label_x + 0.5, label_y + 0.5, f"{r}mm", color='gray', fontsize=7.5, 
                ha='center', va='center', alpha=0.55)

    # Eksen çizgilerini çiz
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.45)
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.45)
    
    # Hedef merkez noktasını çiz (Burun ucu)
    ax.scatter(0, 0, color='black', marker='+', s=130, linewidths=1.8, label='Hedef Merkez (Burun Ucu)', zorder=5)

    # --- VERİ NOKTALARININ VE GRUP MERKEZLERİNİN ÇİZİLMESİ ---
    colors = {1.0: '#2ecc71', 2.0: '#3498db', 3.0: '#e74c3c'}  # Yeşil (1m), Mavi (2m), Kırmızı (3m)
    labels = {1.0: '1.0m Ölçümleri', 2.0: '2.0m Ölçümleri', 3.0: '3.0m Ölçümleri'}
    
    centroids = []
    distances = sorted(list(laser_test_data.keys()))

    for dist in distances:
        coords = laser_test_data[dist]
        x_vals = coords[:, 0]
        y_vals = coords[:, 1]
        
        # Merkez ağırlık noktasını hesapla (Centroid)
        centroid_x = np.mean(x_vals)
        centroid_y = np.mean(y_vals)
        centroids.append((centroid_x, centroid_y))
        
        # A. Tekil deneme noktalarını çiz (Küçültülmüş boyut: 35)
        ax.scatter(x_vals, y_vals, color=colors[dist], alpha=0.45, s=35, edgecolors='none', zorder=3)
        
        # B. Ortalama sapma noktasını çiz (Küçültülmüş boyut: 90)
        ax.scatter(centroid_x, centroid_y, color=colors[dist], edgecolors='black', 
                   s=90, marker='o', linewidths=1.2, label=f'{labels[dist]} Ortalaması', zorder=4)

    # --- SİSTEMATİK KAYMA ROTASININ ÇİZİLMESİ ---
    centroid_coords = np.array(centroids)
    # Merkez noktalarını kronolojik olarak birleştiren kesikli çizgi
    ax.plot(centroid_coords[:, 0], centroid_coords[:, 1], color='#34495e', linestyle=':', 
            linewidth=1.8, label='Sistematik Kayma Rotası', zorder=2)

    # Yön gösteren küçük okların çizgiye eklenmesi
    for i in range(len(centroid_coords) - 1):
        x1, y1 = centroid_coords[i]
        x2, y2 = centroid_coords[i+1]
        ax.annotate('', xy=((x1+x2)/2, (y1+y2)/2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color='#34495e', lw=1.2, ls=':'), zorder=2)

    # --- GRAFİK METADATASI VE ESTETİK AYARLAR ---
    ax.set_xlim(-32, 32)
    ax.set_ylim(-32, 32)
    ax.set_xlabel("Yatay Konum Hatası (mm) [Sol (-) / Sağ (+)]", fontsize=9.5, labelpad=8)
    ax.set_ylabel("Dikey Konum Hatası (mm) [Aşağı (-) / Yukarı (+)]", fontsize=9.5, labelpad=8)
    ax.set_title("Fiziksel Lazer Hedefleme Sapması ve Mekanik Eksen Kayması Analizi\n"
                 "(Donanım-Yazılım Entegrasyon Doğrulama Deneyleri)", fontsize=10.5, fontweight='bold', pad=12)
    
    # Akademik gösterge paneli (Küçültülmüş yazı boyutu)
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='lightgray', 
              fontsize=8, labelspacing=0.35, handletextpad=0.5)
    
    # Dairesel referansların bozulmaması için 1:1 ölçek koruması
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    
    # Görseli kaydet
    plt.tight_layout()
    plt.savefig('lazer_sapma_analizi.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    generate_deviation_plot()