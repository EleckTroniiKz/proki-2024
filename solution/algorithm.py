from PIL import Image
from matplotlib import patches
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import cairosvg


def is_svg(file_path):
    # Überprüfen, ob die Dateiendung .svg ist
    if not str(file_path).lower().endswith('.svg'):
        print("Die Datei ist keine SVG-Datei (basierend auf der Erweiterung).")
        return False
    return True


def findCenterOfGripper(image_path, greifer_path):  

    if is_svg(greifer_path):

        teil_img = Image.open(image_path).convert("L")
        width, height = teil_img.size
        
        gp = str(greifer_path)
        cairosvg.svg2png(url=gp, write_to='svgToPng_image.png',  output_width=width, output_height=height)

        teil_img = Image.open(image_path).convert("L")
        greifer_img = Image.open('svgToPng_image.png').convert("L")
    else:
         # Bild laden und binäre Masken erstellen
        # in Graustufen aufteilen
        teil_img = Image.open(image_path).convert("L")
        greifer_img = Image.open(greifer_path).convert("L")
    
    width, height = teil_img.size
    grip_width, grip_height = greifer_img.size


    greifer_position = calcCenterOfGripper(teil_img, greifer_img)

    #Enable gripper overlap with part(1/2)
    #fig, ax = plt.subplots()
    #ax.imshow(teil_img, cmap="gray")
    #ax.invert_yaxis()

    # Füge das Gripper-Bild als Overlay hinzu
    if greifer_position is not None:
        x, y, z= greifer_position
    else:
        x, y, z = 100, 200, 0
    

    #rotated_gripper = greifer_img.rotate(z, resample=Image.BICUBIC, center=center)
    
    
    #ax.imshow(greifer_img, extent=(x - (grip_width / 2), x + (grip_width / 2), y - (grip_height / 2), y + (grip_height / 2)) , alpha=0.5)

    #Enable gripper overlap with part(2/2)
    #plt.gca().invert_yaxis()
    #plt.title("Part Image with Gripper Overlay")
    #plt.show()



    #fig, ax = plt.subplots()
    #ax.imshow(teil_img, cmap="gray")
    #ax.imshow(greifer_img, cmap="jet", alpha=0.5)
    #plt.title("Part Image with Gripper Overlay")
    #plt.show()


def calcCenterOfGripper(teil_img, greifer_img):
    
    teil_maske = np.array(teil_img) < 140
    greifer_maske = np.array(greifer_img) < 10

    plt.imshow(teil_maske, cmap='gray')
    plt.title('Binary Mask')
    plt.show()

    # Mittelpunkt des Teils berechnen
    # shape = Dimensionen des Bildes
    h, w = teil_maske.shape
    x_part, y_part = w // 2, h // 2

    beste_distanz = float("inf")
    beste_position = None

    # Raster-Suche
    for y in range(0, h, 50):
        for x in range(0, h, 50):
            for winkel in range(0, 360, 45):
                rotate(greifer_maske, 45)
                if not kollisionsprüfung(x, y, winkel, teil_maske, greifer_maske, h, w):
                    distanz = (x - x_part)**2 + (y - y_part)**2
                    if distanz < beste_distanz:
                        beste_distanz = distanz
                        beste_position = (x, y, winkel)
    print(beste_position)
    return beste_position
            


def kollisionsprüfung(
    x, y, winkel, teil_maske: np.ndarray, greifer_maske: np.ndarray, h: int, w: int
) -> tuple[float, float, float]:
    # Greifer rotieren
    rot_greifer = rotate(greifer_maske, winkel, reshape=True)
    gh, gw = rot_greifer.shape

    # Grenzen überprüfen
    if y + gh > h or x + gw > w:
        return True
    
    # Überlappung prüfen
    teil_ausschnitt = teil_maske[y:y+gh, x:x+gw]
    return np.any(np.logical_and(teil_ausschnitt, rot_greifer))
