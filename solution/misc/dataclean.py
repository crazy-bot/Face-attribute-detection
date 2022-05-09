import mediapipe as mp
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
import os
import pandas as pd
import sys
import torch
sys.path.append('/data/suparna/workspace/')
from facenet_pytorch import MTCNN, InceptionResnetV1

def compare_face_models(face_mesh, face_cascade, mtcnn):
    bad_images = []
    for i, row in enumerate(att_df):
        frame_path = 'Tiny_Portrait_%06d.png'%row[0]
        frame = cv2.imread(os.path.join(thumbnail_directory, frame_path))
        
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            # check the images that mediapipe could not detect a face
            print(frame_path)
            bad_images.append(frame_path)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.savefig('frame.png')
            
            #second filter by haarcascaded + dlib        
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            # for (x,y,w,h) in faces:
            #     img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            #     roi_gray = gray[y:y+h, x:x+w]
            #     roi_color = frame[y:y+h, x:x+w]
            print('dllib--',faces)

            # second filter by mtcnn
            # Detect faces
            boxes, _ = mtcnn.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            print('mtcnn---',boxes)


def filter_images(mtcnn):
    bad_images = []
    for i, row in enumerate(att_df):
        frame_path = 'Tiny_Portrait_%06d.png'%row[0]

        frame = cv2.imread(os.path.join(thumbnail_directory, frame_path))
        # second filter by mtcnn
        # Detect faces
        boxes, _ = mtcnn.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #print('mtcnn---',boxes)
        if isinstance(boxes, type(None)):
            print(row)
            bad_images.append(frame_path)
            #plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            #plt.title('my picture')
            #plt.show() 
            
    bad_images = np.array(bad_images)
    np.savetxt('bad_images.txt', bad_images, '%s')

def load_models():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    mtcnn = MTCNN(keep_all=True, device=device)
    

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.1
                )
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    
def visualize(ncols, att_df, column_no):
    vals = np.unique(att_df[:, column_no]) 
    _,axs = plt.subplots(len(vals), ncols, subplot_kw={'xticks': [], 'yticks': []}) 
    
    for j, v in enumerate(vals):
        idxs = att_df[att_df[:,1] == v][:,0]
        idxs = np.random.choice(idxs, 5)
        for i in range(5):
            frame_path = 'Tiny_Portrait_%06d.png'%idxs[i]
            frame = cv2.imread(os.path.join(thumbnail_directory, frame_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axs[j, i].imshow(frame)
            axs[j, i].set(ylabel=v)
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

if __name__ == '__main__':

    thumbnail_directory = '/data/suparna/workspace/TinyPortraits_thumbnails/'
    attribute_file = 'Tiny_Portraits_Attributes.csv'
    att_df = pd.read_csv(attribute_file, header = 0, keep_default_na = False)
    image_index = att_df.index
    att_df = att_df.to_numpy()

    #visualize(5, att_df, 1)
    visualize(5, att_df, 2)
    #visualize(5, att_df, 3)