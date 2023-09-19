# importar a biblioteca do opencv
import cv2 as cv

# ler uma imagem de teste (ajustar o nome e caminho da imagem a gosto
# e garantir que a imagem está no sitio certo
img = cv.imread('images/bg3.jpg', 1)

# mostrar a imagem numa janela intitulada "Imagem de Teste".
cv.imshow('Image', img)

# colocar em ciclo infinito, à espera que se carregue numa tecla
cv.waitKey(0)

# destruir a janela criada
cv.destroyAllWindows()
