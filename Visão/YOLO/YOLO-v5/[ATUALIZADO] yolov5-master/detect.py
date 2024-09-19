#Bibliotecas ______________________________________________________________________________________________________________________________

import argparse
import time     #serve pra ter funções relacionadas ao tempo, como medir a duração de execução de um código, obter o horário atual, etc
from pathlib import Path  #manipula caminhos de arquivos e diretórios 

import cv2  #aqui é usado principalmente as funções para ler, escrever e manipular imagens e vídeos
import torch  #pelo q eu entendi, aqui usa principalmente pra carregar o modelo classificador e pra otimizar o processamento
import torch.backends.cudnn as cudnn
from numpy import random  #nao sei pq o cara botou essa biblioteca, ele só usa em um lugar, e nao serve pra NADA

#____________________________________________________________________________________________________________________________________________





# Outros codigos dentro da pasta que estão sendo referenciados______________________________________________________________________________

from models.experimental import attempt_load #carrega o modelo
from utils.dataloaders import LoadStreams, LoadImages #pra lidar com as imagens ou os videos em tempo real
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_boxes, xyxy2xywh, strip_optimizer, set_logging, increment_path  #a parte visual (imagem, classificar os objetos e tals)

from utils.plots import plot_one_box, dominant_color, calculate_general_center #O CODIGO MAIS IMPORTANTE É ESSE "plots.py", basicamente é ele que determina a cor que o modelo vai ler, além de ser ele que vai determinar as coordenadas do objeto
#recomendo fortemente que analise o detect.py enquanto deixa o plots.py aberto

from utils.torch_utils import select_device, reshape_classifier_output, time_sync #ajuda na classificação tbm

#_____________________________________________________________________________________________________________________________________________


#Principal função do codigo, é aqui que a gente vai detectar as coisas________________________________________________________________________
def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size #permite a gente escolher os pesos, as imagens/videos etc.
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # diretorios
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run (que é quando a gente inicia o codigo pelo terminal com aquele comando lá "python detect.py....")
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True) 

    # inicia a detecção
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu' #se for GPU 

    # carregar modelo
    model = attempt_load(weights, device=device) 
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)  # checar o tamanho da imagem
    if half:
        model.half()

    # classificação em duas etapas (classifica primeiro e depois volta aqui e classifica de novo pra ter certeza)
    classify = False
    if classify:
        modelc = reshape_classifier_output(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # aqui a gente configura qual tipo de dado estamos usando (câmera, imagem, video etc)
    vid_path, vid_writer = None, None
    
    #se estamos usando a webcam:
    if webcam: 
        view_img = check_imshow() # verifica se a exibição de imagem está disponível
        cudnn.benchmark = True # otimiza o desempenho em GPU
        dataset = LoadStreams(source, img_size=imgsz, stride=stride) # carrega dados de vídeo
    
    #senão, se estamos usando imagens
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # aqui a gente pega os nomes usados pelo modelo e (teoricamente) as cores, mas nao entendi o propósito dessa variável cores (ele nem usa no codigo)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # aqui começa a rodar a inferência ou seja a detecção em si
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time() # marca o tempo inicial
    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # inferencia
        t1 = time_sync() # marca o tempo antes da inferência

        pred = model(img, augment=opt.augment)[0] # executa a inferência no modelo

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_sync() # marca o tempo após a inferência

        # aplica a primeira classificação
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # processa as detecções
        for i, det in enumerate(pred):  # aqui ele enumera o número de detecções por imagem
            if webcam:  # se estiver usando a webcam
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:  # se não estiver, salva a imagem ou video num diretorio especifico
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # redimensiona as caixas de detecção do tamanho da imagem para o tamanho original da imagem
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

                # imprime resultados
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # salva resultados
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # se deve salvar os resultados em um arquivo
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # se deve salvar ou exibir a imagem
                        color, color_val, h = dominant_color(xyxy, im0) # obtém a cor dominante da caixa (pela função no plots.py)
                        label = f'{color} {names[int(cls)]}'
                        plot_one_box(xyxy, im0, label=label, color=color_val, line_thickness=3) # desenha a caixa e o rótulo na imagem
                        calculate_general_center(im0) # desenha o circulo com as coordenadas gerais


            # imprime o tempo gasto na inferencia
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # exibe os resultados
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # salva os resultados (se forem imagens)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path: 
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")  #mensagem ao salvar imagens ou video em algum diretorio

    print(f'Done2. ({time.time() - t0:.3f}s)') #tempo de inferencia ao salvar imagens ou video


if __name__ == '__main__':
   
    #configura oq a gente pode escrever na linha de comando (no terminal)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
