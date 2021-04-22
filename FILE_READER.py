# from FIle_open import file_select
import cv2 as cv
import os
import streamlit as st
from PIL import Image
import glob

def image_length_checker(cwd):
    train_data_length = 0
    validation_data_length = 0
    folders_check = ['train','validation']
    folders = []
    for i in folders_check:
        
        for _, dirnames,_ in os.walk(cwd+'/data/'+str(i)):
            folders.append(dirnames)
        if len(folders[0]) >=2:
            for j in folders[0]:
                # length = 0
                
                # st.text(i)   
                for _, dirnames,b in os.walk(cwd+'/data/'+str(i)+'/'+str(j)):
                    # length.append(b) 
                    # st.text(f'{j} has {len(b)}')
                    if i == 'train':
                        if (len(b)<70):
                            st.warning(f'Model need atleast 90 images, put more images in {j} folder')
                            st.stop()
                        else:
                            train_data_length = train_data_length+int((len(b)))
                    elif i == 'validation':
                        if (len(b)<15):
                            st.warning(f'Model need atleast 90 images, put more images in {j} folder')
                            st.stop()
                        else:
                            validation_data_length = validation_data_length+int(len(b))
        else:
            st.warning('Please Make atleast two label')
            st.stop()
    return train_data_length, validation_data_length

def folder_T_V_maker (cwd,images_folder_name):
    # st.text('dd')
    # st.text(images_folder_name)
    folders = 0
    folder_name = ['None']
    for _, dirnames,_  in os.walk(cwd+'/data/'+str(images_folder_name)):
        folder_name += dirnames
        folders += len(dirnames)
    # st.text('dd')    
    # st.text(folders)
    # if (folders) >=2:
    option = st.sidebar.selectbox('Check Created Lables and images'\
        ,(folder_name))
    if option != "None": 
        image_show(option,cwd,images_folder_name)
    else:
        st.text(' ')
    # else:
    #     st.sidebar.info('Please Make atleast two label')
    #     st.stop()
    return folders

def opener(cwd,option,arr,images_folder_name):
    for i in arr:
        image = Image.open(os.path.join(cwd+'/data/'+str(images_folder_name)+'/'+str(option)+'/'+i))
        st.image(image, caption=option,width=150)
 
def image_show(option,cwd,images_folder_name):
    arr = os.listdir(os.path.join(cwd+'/data/'+str(images_folder_name)+'/'+str(option)+'/'))
    opener(cwd,option,arr,images_folder_name)


def img_copy(path,cwd,data_folder,Label):
    for name,img in enumerate(path):
        # st.text(img)
        image = cv.imread(img)
        # image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        image = cv.resize(image, (224, 224))
        cv.imwrite(os.path.join(cwd+'/data/'+str(data_folder)+'/'+str(Label))+'/'+str(name)+'.jpg',image)

def test_img_copy(test_path,cwd):
    for name,img in enumerate(test_path):
        # st.text(img)
        image = cv.imread(img)
        # image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        image = cv.resize(image, (224, 224))
        cv.imwrite(os.path.join(cwd+'/data/test/test/')+str(name)+'.jpg',image)

def result_output_show(cwd,result,label_map):
    
    test_arr_label = os.listdir(os.path.join(cwd+'/data/train/'))
    test_images_arr = os.listdir(os.path.join(cwd+'/data/test/test/'))
    st.warning('Final Prediction')
    for i,j in zip(result,test_images_arr):
        TEST_image = Image.open(os.path.join(cwd+'/data/test/test/'+str(j)))
        # st.text(output_label[i])
        st.image(TEST_image, caption=label_map[i],width=150)
        st.info('  ')
def label_folder_maker(cwd):
    Label = st.sidebar.text_input('Enter Label')
    if Label:
        # label_list = []
        if not os.path.exists(cwd+'/data/train/'+str(Label)):
            try:
                # st.text('entered')
                # label_list = label_list.append(Label)
                os.mkdir(cwd+'/data/train/'+str(Label))  
            except OSError as error: 
                print("Directory '%s' can not be created" % directory) 
        if not os.path.exists(cwd+'/data/validation/'+str(Label)):
            try:
                # label_list = label_list.append(Label)
                os.mkdir(cwd+'/data/validation/'+str(Label))  
            except OSError as error: 
                print("Directory '%s' can not be created" % directory)        
# def data_division(cwd,Label):
#     for _, dirnames,_  in os.walk(cwd+'/data/train/'+str(images_folder_name)):



