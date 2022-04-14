import os
import ntpath
import torch
import cv2
from PIL import Image
from tools import transforms

from tools.model import get_model_object_detection

def pred(path_model, path_test_img):
    #PATH = "/opt/data/team/hien/models/model_01_03_100.pt"
    filename = path_test_img #"test_mid.png"
    #pred_path= root_dir + "/files/prediction/" + filename
    
    model = get_model_object_detection(4)
    model.eval()
    device = torch.device('cpu')
    model.load_state_dict(torch.load(path_model))
    model.to(device)
    #get item
    image= Image.open(path_test_img).convert('RGB')
    img_np = cv2.imread(path_test_img)
    convert_tensor = transforms.ToTensor()
    image_tensor, _ = convert_tensor(image, 0)


    #generate prediction
    with torch.no_grad():


        prediction = model([image_tensor.to(device)])
        print("Hier ist die Prediction")

        pred = prediction[0]['boxes']
        print(len(pred))
        pred_img = pred.cpu().numpy()


        for i in range(len(pred_img)-1):
            x = pred_img[i][0]
            y = pred_img[i][1]
            x2 = pred_img[i][2]
            y2 = pred_img[i][3]

            startpoint = (int(x),int(y))
            endpoint= (int(x2), int(y2))

            img_np = cv2.rectangle(img_np, startpoint, endpoint, (0,0,255), 2)

        directory = '/opt/data/team/hien/models'
        os.chdir(directory)
        cv2.imwrite("pred_"+ ntpath.basename(path_test_img), img_np)



    print("done")