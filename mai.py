import cv2
import numpy as np

img = cv2.imread("img/4.bmp").astype(np.float32) / 255.0
fundo = cv2.imread("lgd.jpg").astype(np.float32) / 255.0

alt, larg = img.shape[:2]
# da um resize baseado na img
fundo = cv2.resize(fundo, (larg, alt), interpolation=cv2.INTER_AREA)

"""
parametros do verde no HSL
"""
H_min = 40
H_max = 80
S_min = 60


def criar_mask_verde(imgg):
    """
    a funcao calcula o quanto que o verde domina e calcula se ele o nivel de opacidade/translucidez
    """
    # separa canal
    R, G, B = imgg[:, :, 2], imgg[:, :, 1], imgg[:, :, 0]

    # calculando o maior
    maior_RB = 0.5 * (R + B + np.abs(R - B))
    verdice = G - maior_RB
    verdice = (verdice - 0.05) / (0.2)
    verdice = np.clip(verdice, 0, 1)

    mascara = 1 - verdice
    return mascara


def reduzir_verde(imgg):

    img = cv2.cvtColor((imgg * 255).astype(np.uint8), cv2.COLOR_BGR2HLS)

    H, L, S = cv2.split(img)
    mask_Verde = (H >= H_min) & (H <= H_max) & (S > S_min)
    # saturacao do verde em 0, deixando ele cinza
    S[mask_Verde] = 0
    S = np.clip(S, 0, 255)

    # junta os canais
    HLS_merge = cv2.merge([H, L, S])
    # conversao
    return cv2.cvtColor(HLS_merge, cv2.COLOR_HLS2BGR).astype(np.float32) / 255.0


if __name__ == "__main__":

    # cria a mascara do objeto
    mascara_objeto = criar_mask_verde(img.astype(np.float32))
    # reduz o verde da imagem
    img_sem_verde = reduzir_verde(img)
    # borra
    mascara_objeto = cv2.GaussianBlur(mascara_objeto, (5, 5), 0)
    # aplica a imagem sem verde no objeto
    objeto_corrigido = img_sem_verde * mascara_objeto[..., None]
    # aplica o objeto no fundo
    resultado = objeto_corrigido + fundo * (1 - mascara_objeto[..., None])

    cv2.imshow("Resultado final", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
