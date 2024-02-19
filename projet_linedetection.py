import numpy as np
import cv2 as cv



def draw_lines(img, lines, color=[0, 255, 0], thickness=2):
    if lines is None:
        return img

    # Calculer la pente et l'ordonnée à l'origine de chaque ligne
    lines_data = [(x1, y1, x2, y2, (y2-y1)/(x2-x1), y1 - (y2-y1)/(x2-x1)*x1) for line in lines for x1,y1,x2,y2 in line]
    
    # Seuil pour regrouper les lignes
    slope_threshold = 0.1
    intercept_threshold = 10.0

    # Regrouper les lignes
    grouped_lines_data = []
    for line_data in lines_data:
        x1, y1, x2, y2, slope, intercept = line_data
        for group in grouped_lines_data:
            if abs(group[0] - slope) < slope_threshold and abs(group[1] - intercept) < intercept_threshold:
                group[2].append(line_data)
                group[0] = np.mean([line[4] for line in group[2]])  # Update average slope
                group[1] = np.mean([line[5] for line in group[2]])  # Update average intercept
                break
        else:
            grouped_lines_data.append([slope, intercept, [line_data]])

    # Dessiner les lignes moyennes
    for group in grouped_lines_data:
        slope, intercept, _ = group
        y1_new = img.shape[0]  # le bas de l'image
        y2_new = int(y1_new * 0.6)  # un peu en dessous du milieu de l'image
        x1_new = int((y1_new - intercept) / slope) if slope != 0 else 0
        x2_new = int((y2_new - intercept) / slope) if slope != 0 else 0
        cv.line(img, (x1_new, y1_new), (x2_new, y2_new), color, thickness)

    return img

# Dans votre boucle principale, remplacez le dessin des lignes par la nouvelle fonction

def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)

    # Définir les coordonnées du rectangle de recadrage (x, y, largeur, hauteur)
    x, y, w, h = 350, 450, 600, 600  # Ajustez ces valeurs en fonction de vos besoins

    rectangle = np.array([[
        (x, y),
        (x + w, y),
        (x + w, y + h),
        (x, y + h)
        ]])

    cv.fillPoly(mask, rectangle, 255)
    masked_image = cv.bitwise_and(img, mask)
    return masked_image



def color_filter(image):
    # Convertir l'image en HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Définir les plages de couleurs pour le blanc et le jaune
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 255, 255])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    # Créer des masques pour le blanc et le jaune
    white_mask = cv.inRange(hsv, white_lower, white_upper)
    yellow_mask = cv.inRange(hsv, yellow_lower, yellow_upper)

    # Combiner les masques
    mask = cv.bitwise_or(white_mask, yellow_mask)
    filtered = cv.bitwise_and(image, image, mask=mask)
    return filtered


# Créer un objet VideoCapture
video_path = "/Users/tourealkaya/Downloads/project_video.mp4"
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print("Erreur lors de l'ouverture de la vidéo")

fps = cap.get(cv.CAP_PROP_FPS)
wait_time = int(1000/fps)

while cap.isOpened():
    # Lire la vidéo frame par frame
    ret, frame = cap.read()
    if ret:
        #filtrer les couleurs
        filtered = color_filter(frame)
        
        #convertir en ndg
        gray = cv.cvtColor(filtered, cv.COLOR_BGR2GRAY)
        
        #flou gaussien 
        blur = cv.GaussianBlur(gray, (5,5), 0)
        
        #detection de bord canny
        edges = cv.Canny(blur, 50, 150)
        
        #applique la region d'interet
        cropped_edges = region_of_interest(edges)
        
        #transformation de Hough
        lines = cv.HoughLinesP(cropped_edges, 1, np.pi/180, 90, minLineLength=10, maxLineGap = 5)
        
        frame = draw_lines(frame, lines)
       
        # Afficher le frame
        cv.imshow('Video', frame)

        # Quitter si 'q' est pressé
        if cv.waitKey(wait_time) & 0xFF == ord('q'):
            break
    else:
        break

# Libérer les ressources et fermer les fenêtres
cap.release()