para configurar teclado na versão PT-BR precisamos:
escreva no terminal: localectl status
Observe como está seu VC Keymap, caso esteja aparecendo n/a isso significa que seu teclado realmente não está configurado para nenhum modelo
Após isso escreva no terminal:
sudo dpkg-reconfigure keyboard-configuration
URDF - Unified Robotics Description File


# URDF - Unified Robotics Description File

Aqui será apresentado um breve resumo sobre URDF e um tutorial de como escrever um arquivo URDF. Primeiramente, **URDF** significa **Unified Robot Description Format** (Formato Unificado de Descrição de Robôs). É um formato **XML** (extensible markup language, ou linguagem de marcação extensível) usado no **ROS** (Robot Operating System) para descrever a estrutura física e os componentes de um robô. Um arquivo URDF define as várias partes de um robô (chamadas de _links_) e como elas se conectam (_joints_), além de propriedades físicas como tamanho, peso e aparência.
>**Lembrete:** para visualizar seu robô, é necessário que o ROS2 esteja instalado corretamente. **Este tutorial tem como referência a versão ROS2 Humble Hawksbill.**

Sumário:
-
-  [Principais componentes de um arquivo URDF](#mainparts)
 - [Links](#whatisalink)
 - [Joints (Juntas)](#whatisajoint)
- Executando e visualizando arquivos `.urdf` no ROS2
- Projeto exemplo

<div id='mainparts'/>  

# Principais componentes de um arquivo URDF

Um arquivo URDF é escrito em XML, uma linguagem de marcação usada para estruturar dados de forma hierárquica. A estrutura geral do URDF permite que você descreva os componentes do robô e suas interações.

- [Links](#whatisalink)
- [Joints (Juntas)](#whatisajoint)

****

## Exemplo de código URDF
	
``` xml
<?xml version="1.0"?> <!-- declaração da versão de XML utilizada -->
<robot name="robot_name"> <!-- define o nome do robô; evite espaços e caracteres especiais -->
  <link name="link_name"> <!-- primeiro link do robô. Os parâmetros do link serão abordados na seção seguinte -->
    <visual> <!-- define os parâmetros visuais, geométricos e estéticos -->
    </visual>
    
    <collision>
    </collision>
    
    <inertial> <!-- define parâmetros como massa para fins de simulação -->
    </inertial>
  </link> <!-- encerra a descrição do link -->
  
  <joint> <!-- declara uma junta -->
    <axis .../>
    <parent link .../>
    <child link .../>
    <origin .../>
  </joint> <!-- encerra a descrição da junta -->
</robot> <!-- encerra a descrição do robô -->
```

<div id='whatisalink'/>  

## Links

Os _links_ indicam as partes do robô, tais como partes móveis, como braços,  ou até mesmo sensores ou rodas. Cada link possui as propriedades:

-   **visual:** descreve a geometria do link e suas características visuais e estéticas.
-   **collision:** define características para detectar colisões, essenciais para simulações em ambientes como o Gazebo.
-   **inertial:** adiciona propriedades inerciais, como massa e centro de gravidade, importantes para simulação física.

``` xml
<?xml version="1.0"?>
<robot name="multipleshapes">
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </visual>
  </link>

  <link name="right_leg">
    <visual>
      <geometry>
        <box size="0.6 0.1 0.2"/>
      </geometry>
    </visual>
  </link>

  <joint name="base_to_right_leg" type="fixed">
    <parent link="base_link"/>
    <child link="right_leg"/>
  </joint>

</robot>
```
Nesse exemplo, o robô possui uma base em formato de cilindro e uma "perna" em formato de caixa.


<div id='whatisajoint'/>  

## Joints

As _joints_ conectam os _links_ do robô, definindo como eles interagem e se movem entre si. Existem diferentes tipos de juntas, incluindo:

-   **fixed joint:** junta fixa, sem movimento entre os links conectados.
-   **revolute joint:** permite rotação limitada em torno de um eixo.
-   **continuous joint:** permite rotação contínua em torno de um eixo, sem limite de ângulo.
-   **prismatic joint:** permite movimento linear ao longo de um eixo.
-   **floating joint:** permite seis graus de liberdade (três rotações e três translações).
-   **planar joint:** permite movimento livre em um plano 2D.

``` xml
<?xml version="1.0"?>
<robot name="multipleshapes">
</robot>
```
##
# Sites úteis 
- [Site oficial ROS2 Humble](https://docs.ros.org/en/humble/)
- [Documentação ROS2 Humble -> arquivos URDF](https://docs.ros.org/en/humble/Tutorials/Intermediate/URDF/URDF-Main.html)
