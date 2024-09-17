# Funções do OpenCV

## Trackbar
#### cv2.createTrackbar() 
```python 
cv2.createTrackbar()
    NOME DO SLIDER : str ,      #Nome que parecerá ao lado do "slider"
    NOME DA JANELA : str ,      #Nome da janela em que o "slider" estará
    POSIÇÂO INICIAL : int ,     #Posição inicial que o "slider" estará 
    ALCANCE DOS VALORES : int , #Todos os valores que podem aparecer, funciona como "range(0,n+1)"
    FUNÇÃO : (int) -> None      #Toda vez que o "slider" mover essa função é chamada
) 
```
- O parâmetro ```FUNÇÃO``` recebe uma váriavel inteira que corresponde a posição do _slider_ ;
- A função retorna ```None``` .

#### cv2.getTrackbarPos()
```python
cv2.getTrackbarPos(
    NOME DO SLIDER : str    #Nome do "slider" desejado
    NOME DA JANELA : str    #Nome da janela em que o "slider" está
)
```
- A função retorna ```int``` que é a posição do slider.