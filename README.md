# image-de-blurring

O projeto desenvolvido trata-se de image de-blurring via minimização de quadrados mínimos. A partir de uma imagem preto e branco, produz-se a versão borrada da imagem usando um filtro gaussiano 2D e em seguida desborra-se a imagem por meio do método proposto.

# Convolução
No processamento de imagens, um kernel, matriz de convolução ou máscara é uma matriz pequena. Ele é usado para desfoque, nitidez, relevo, detecção de bordas e muito mais. Isso é feito fazendo uma convolução entre um kernel e uma imagem. Um exemplo simples de convolução trata-se da máscara para detecção de bordas na imagem.

#### Detecção de bordas por meio de máscaras de Sobel
Para obter a máscara, considera-se que cada vetor ortogonal é uma estimativa derivativa direcional multiplicada por um vetor unitário especificando a direção da derivada. A soma vetorial dessas estimativas de gradiente simples equivale a uma soma vetorial dos 8 vetores derivativos direcionais. Assim, um ponto na grade cartesiana e seus oito valores de densidade vizinhos, como mostrado:

|  |  |  |
|--|--|--|
| a | b | c |
| d | e | g |
| h | i | j |

gradiente = G = (c - g)[1, 1] + (a - i)[-1, 1] + (b - h)[0, 1]

Normalizando,
2G = (c - g)[1, 1] + (a - i)[-1, 1] + (b - h)[0, 1]

Logo,
2G = [(c - g - a + i) + 2(f - d), (c - g + a - i) + 2(b - h)]

Assim, para bordas a partir de gradientes na direção horizontal, tem-se a seguinte máscara

|  |  |  |
|--|--|--|
| -1 | 0 | 1 |
| -2 | 0 | 2 |
| -1 | 0 | 1 |

já para bordas a partir de gradientes na direção vertical, tem-se a seguinte máscara

|  |  |  |
|--|--|--|
| 1 | 2 | 1 |
| 0 | 0 | 0 |
| -1 | -2 | -1 |

Implementação:

```python
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
import numpy as np
import matplotlib.cm as cm
from PIL import Image


im = Image.open('assets/pb.png')
im_grey = im.convert('L') # converter para preto e branco
im_array = np.array(im_grey)
plt.imshow(im_array, cmap=cm.Greys_r)
plt.savefig('assets/edge_before.png')


kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], np.float32)
result = convolve2d(im_array, kernel)
plt.imshow(result, cmap=cm.Greys_r)
plt.savefig('assets/edge_after.png')
```

Data a imagem a seguir,

![edge_before](assets/edge_before.png)

Para a detecção de bordas a partir de gradientes na vertical, obtém-se

![edge_after](assets/edge_after.png)


### Referências

- Sobel Edge Detection Algorithm

https://pdfs.semanticscholar.org/6bca/fdf33445585966ee6fb3371dd1ce15241a62.pdf
