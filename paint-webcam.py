import cv2
import numpy as np
from time import gmtime, strftime

COLORS = [(0,0,255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
ERASER = (0,0,0)
COLOR_IDX = 0

# Função que gera a mascara de cor
# Essa função foi adaptada desse projeto: https://towardsdatascience.com/tutorial-webcam-paint-opencv-dbe356ab5d6c
def color_mask(frame_hsv, lower, upper):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.inRange(frame_hsv, lower, upper)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def paint_screen(frame, tela):
    # Convertendo em grayscale
    gray = cv2.cvtColor(tela, cv2.COLOR_BGR2GRAY)
    # Fazendo threshold inverso p deixar a tinta preta e o fundo branco
    _, thr = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
    # Convertendo o threshold em rgb pra poder aplicar como mascara no frame
    rgb_thr = cv2.cvtColor(thr, cv2.COLOR_GRAY2RGB)
    # Aplicando o threshold como mascara
    mask_screen = cv2.bitwise_and(frame, rgb_thr)
    # Aplicando OR entre o frame mascarado e a tela
    return cv2.bitwise_or(mask_screen, tela)

def change_color():
    return COLORS[(COLOR_IDX+1)%len(COLORS)]


# Cor atual (sempre inicia com vermelho (BGR))
current_color = COLORS[COLOR_IDX]

# Tons azuis
lower = np.array([110,50,50])
upper = np.array([130,255,255])

# Dimensões da tela do meu celular
h, w = 1080, 1920

# Configurando o video da camera do meu celular
video = cv2.VideoCapture(0)
cam_ip = "https://10.0.0.104:8080/video"
video.open(cam_ip)

# Criando a tela de desenhos
tela = np.zeros((h//2, w//2, 3), np.uint8)

# Último ponto detectado sendo None, pois ainda não começou a detecção
last_center = None

while(True):
    ret, frame = video.read()
    frame_rsz = cv2.resize(frame, (w//2, h//2)) 
    frame_hsv = cv2.cvtColor(frame_rsz, cv2.COLOR_BGR2HSV)

    # Cria a mascara que isola o que for azul
    mask_color = color_mask(frame_hsv, lower, upper)

    # Pega os contornos dos azuis detectados
    cnts, _ = cv2.findContours(mask_color.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # caso tenha detectado e não seja None
    if cnts is not None and len(cnts) > 0:
    	# Ordena e pega o maior
        cnt = sorted(cnts, key = cv2.contourArea, reverse = False)[0]
        # Gera o circulo ao redor da area do contorno
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        # Desenha o circulo aop redor da area detectada
        cv2.circle(frame_rsz, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        # Csa os "momentos" para calcular o centro do circulo
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        # Se não houver um centro detectado anterior, atribui o mesmo
        if last_center is None: last_center = center
        # Desenha a linha
        cv2.line(tela, last_center, center, current_color, 10)
        last_center = center
    else:
        # paro caso de parar de detectar centros, não puxar a linha a partir do ultimo
        last_center = None

    # Adicionando o indicador de qual a cor atual selecionada
    frame_rsz = cv2.rectangle(frame_rsz, (40,20), (140,65), current_color, -1)
    cv2.putText(frame_rsz, "Cor atual", (49, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    final_screen = paint_screen(frame_rsz, tela)
    cv2.imshow('Paint Webcam', final_screen)
    

    # Comando de Teclado
    k = cv2.waitKey(1)

    # Para sair, aperte q, a imagem sera salva qnd fechar
    if  k == ord('q'):
        output_name = 'paint_' + strftime("%Y%m%d_%H%M%S", gmtime()) +".jpg"
        cv2.imwrite(output_name, final_screen)
        break

    # para salvar a imagem, aperte s
    if  k == ord('s'):
        output_name = 'paint_' + strftime("%Y%m%d_%H%M%S", gmtime()) +".jpg"
        cv2.imwrite(output_name, final_screen)

    # para ativar a borracha, aperte e
    if  k == ord('e'):
        current_color = ERASER

    # Para trocar a cor atual, aperte barra de espaço
    if  k == 32:
        current_color = change_color()
        COLOR_IDX = (COLOR_IDX+1)%len(COLORS)

    # Para limpar a tela, aperte c
    if  k == ord('c'):
        tela = np.zeros((h//2, w//2, 3), np.uint8)


video.release()
cv2.destroyAllWindows()