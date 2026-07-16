# assets/media/lock_on_bar_plot.py

import numpy as np
import matplotlib.pyplot as plt

# Deneysel verilerin tanımlanması (Merkez noktası 0sn olduğu için grafiğe dahil edilmemiştir)
konumlar = ['YUKARI', 'AŞAĞI', 'SOL', 'SAĞ']
sureler_2m = [2.55, 2.56, 2.86, 3.63]
sureler_4m = [3.45, 2.40, 2.35, 3.00]

def generate_bar_plot():
    # Yüksek çözünürlüklü grafik alanı oluştur (300 DPI)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    x = np.arange(len(konumlar))  # Konumların X eksenindeki yerleri
    width = 0.35  # Sütun genişliği

    # Sütunları çiz (Yarı saydam ve net renkler)
    rects1 = ax.bar(x - width/2, sureler_2m, width, label='2.0m Mesafe', color='#3498db', edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + width/2, sureler_4m, width, label='4.0m Mesafe', color='#e67e22', edgecolor='black', linewidth=0.8)

    # Eksen ve grafik başlıklarını ekle
    ax.set_ylabel('Kilitlenme Süresi (Saniye)', fontsize=10, fontweight='bold', labelpad=10)
    ax.set_xlabel('Hedefin Kamera Görüş Alanı Sınırlarındaki Konumu', fontsize=10, fontweight='bold', labelpad=10)
    ax.set_title('Kamera Görüş Alanı Sınırlarındaki Hedefe Kilitlenme Süreleri\n'
                 '(Kamera Çözünürlüğü: 1720 x 1080 px | Eksen Limitleri: ±60° Pan, ±40° Tilt)', 
                 fontsize=11, fontweight='bold', pad=15)
    
    # X ekseni etiketlerini ayarla
    ax.set_xticks(x)
    ax.set_xticklabels(konumlar, fontsize=9.5, fontweight='bold')
    
    # Y ekseni limitini ve grid çizgilerini ayarla (Sürelerin rahat görünmesi için)
    ax.set_ylim(0, 5.0)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Her sütunun üzerine saniye değerini yazdır (Okunabilirliği artırmak için)
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f} s',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # Sütunun 3 piksel üzerinde göster
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    # Gösterge paneli (Legend)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='lightgray', fontsize=9)

    # Arka plan çerçevesini hafiflet
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig('kilitlenme_süreleri_analizi.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    generate_bar_plot()