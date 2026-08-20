import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_diagram():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Helper function to draw boxes
    def draw_box(x, y, w, h, text, color='lightblue', fontweight='bold'):
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight=fontweight, wrap=True)

    # 1. ENTRADAS
    draw_box(10, 80, 25, 12, "ENTRADAS\n(Historial 0.5s)", color='#f9f9f9')
    draw_box(10, 70, 25, 8, "LIDAR (360 x 5)", color='#e1f5fe')
    draw_box(10, 62, 25, 8, "Velocidad (2 x 5)", color='#e1f5fe')

    # 2. ENCODERS
    draw_box(45, 75, 20, 10, "PROCESAMIENTO\n(MLP Encoders)", color='#fff9c4')
    draw_box(45, 65, 20, 8, "Positional Encoding", color='#fff9c4')

    # 3. TRANSFORMER
    draw_box(75, 55, 20, 30, "CEREBRO\n(Transformer Encoder)\n\nSelf-Attention\nLayer Norm\nFeed Forward", color='#c8e6c9')

    # 4. SALIDAS
    draw_box(45, 40, 20, 10, "DECODER\n(MLP Regressor)", color='#ffccbc')
    draw_box(10, 40, 25, 10, "SALIDA (t+1)\nComandos [v, w]", color='#ffab91', fontweight='extra bold')

    # Arrows
    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))

    arrow(35, 74, 45, 78) # Entradas to Encoder
    arrow(35, 66, 45, 68) # Entradas to PosEnc
    arrow(65, 75, 75, 75) # Encoder to Transformer
    arrow(85, 55, 65, 45) # Transformer to Decoder
    arrow(45, 45, 35, 45) # Decoder to Output

    plt.title("ARQUITECTURA DEL GENERADOR DE TRAYECTORIAS (TRANSFORMER)", fontsize=14, fontweight='bold', pad=20)
    plt.savefig('diagrama_arquitectura.png', dpi=300, bbox_inches='tight')
    print("Imagen guardada como: diagrama_arquitectura.png")

if __name__ == "__main__":
    draw_diagram()
