import streamlit as st
import os
import io
import numpy as np
from FILE_READER import *
from images_path import img_path
import time
import cv2 as cv
from PIL import Image
import math
import shutil
# from model import train_reading_data
from test import train_reading_data,model_training,test_reading_data,model_prediction


if __name__ == "__main__":
    cwd = os.getcwd()
    # st.error(' ')
    st.header('Making Deep Learning Simple')
    st.error(' ')
    intro_image =Image.open(os.path.join(cwd+'/intro_image.jpg'))
    st.image(intro_image, use_column_width=True)
    # st.header('Making Deep Learning Simple')
    # st.error(' ')
    Lables_list = []
    st.sidebar.subheader('Input Label')
    for _, label_list,_  in os.walk(cwd+'/data/train/'):
        for i in label_list:
            Lables_list.append(label_list)
    # st.text(Lables_list[0])
    if len(Lables_list) == 0:
        label_folder_maker(cwd)
        # Label = st.sidebar.text_input('Enter Label')
        # if Label:
        #     # label_list = []
        #     if not os.path.exists(cwd+'/data/train/'+str(Label)):
        #         try:
        #             # label_list = label_list.append(Label)
        #             os.mkdir(cwd+'/data/train/'+str(Label))  
        #         except OSError as error: 
        #             print("Directory '%s' can not be created" % directory) 
        #     if not os.path.exists(cwd+'/data/validation/'+str(Label)):
        #         try:
        #             # label_list = label_list.append(Label)
        #             os.mkdir(cwd+'/data/validation/'+str(Label))  
        #         except OSError as error: 
        #             print("Directory '%s' can not be created" % directory) 
        
    else:
        st.text('  ')
    
    more_label =  st.sidebar.select_slider('Create More Labels',('none','yes'))
    try:
        if more_label == 'yes':
            label_folder_maker(cwd)
        Lables_list = []
    except Exception:
        st.stop()
    # st.sidebar.subheader('Input Label')
    for _, label_list,_  in os.walk(cwd+'/data/train/'):
        for i in label_list:
            Lables_list.append(label_list)    
    if len(Lables_list) > 0:
        Label = st.sidebar.selectbox('Registered Label',(tuple(Lables_list[0])))
            
        if  Label:
            cwd = os.getcwd()
            st.sidebar.info('Please select images')
            if st.sidebar.button('Select Images'):
                path = img_path()
                train_test_length = math.floor(0.8*len(path))
                # st.text(len(path[:train_test_length]))                
                if path:
                    img_copy(path[:train_test_length],cwd,'train',str(Label))
                    img_copy(path[train_test_length:],cwd,'validation',str(Label))
                    with st.spinner('Thanks for selecting images'):
                        time.sleep(1)
        delete_label = st.sidebar.checkbox('Click to delete Label')
        if delete_label:
            del_Lables_list = []
            # st.sidebar.subheader('Input Label')
            for _, label_list,_  in os.walk(cwd+'/data/train/'):
                for i in label_list:
                    del_Lables_list.append(label_list)
            del_Lables_list = del_Lables_list[0]
            # del_Lables_list = del_Lables_list.append('None') 
            del_label = st.selectbox('Select Label to delete',(del_Lables_list))
            if del_label:
                st.error('Are you sure want to delete')
                if st.button('Delete'):
                    # st.text(os.path.join(cwd+'/data/validation/'+str(del_label)))
                    shutil.rmtree(os.path.join(cwd+'/data/train/'+str(del_label)))
                    shutil.rmtree(os.path.join(cwd+'/data/validation/'+str(del_label)))
                    st.text('done')

            
        agree = st.sidebar.checkbox('Click to Validate Input data')
        if agree: 
            train_data_length, validation_data_length = image_length_checker(cwd)
            # st.text(train_data_length)
            # st.text(validation_data_length)
            data_folder_check = st.sidebar.radio('Select folder for checking uploaded images',('Train_images','Validation_images'))
            if data_folder_check == 'Train_images':
                images_folder_name = 'train'
                NUM_CLASSES = folder_T_V_maker(cwd,images_folder_name)

            if data_folder_check == 'Validation_images':    
                images_folder_name = 'validation'
                NUM_CLASSES = folder_T_V_maker(cwd,images_folder_name)
                st.text(NUM_CLASSES)
            Model = st.sidebar.radio('Select for Making Model',('None','Model','test_images','Prediction'))
            if Model == 'Model':
                if os.path.exists(cwd+'/best.hdf5'):
                    st.warning('Model is already trained')
                    
                    if st.sidebar.button('Want to train again'):
                        with st.spinner('Wait Model is running...'):
                            train_generator,validation_generator,label_map = train_reading_data(cwd) 
                            model_training(cwd,train_generator,validation_generator,NUM_CLASSES,train_data_length,validation_data_length)
                else:
                    with st.spinner('Wait Model is running...'):
                        # st.text(NUM_CLASSES)
                        train_generator,validation_generator,label_map = train_reading_data(cwd) 
                        model_training(cwd,train_generator,validation_generator,NUM_CLASSES,train_data_length,validation_data_length)
            elif Model == 'test_images':
                if not os.path.exists(cwd+'/data/test/test/'):
                    try:
                        os.mkdir(cwd+'/data/test/test/')  
                    except OSError as error: 
                        print("Directory '%s' can not be created" % directory)  
                if st.sidebar.button('Select test images'): 
                    test_path = img_path()
                    if test_path:
                        test_img_copy(test_path,cwd)
                        with st.spinner('Thanks for selecting images'):
                            time.sleep(1)                  
            elif Model == 'Prediction':
                with st.spinner('Wait Model is gearing for Prediction...'):
                    for _, dirnames,b in os.walk(cwd+'/data/test/test'):
                        if len(b) == 0:
                            st.warning('No Image found, please upload')
                            st.stop()                        
                        if len(b)>0:
                            train_generator,validation_generator,label_map = train_reading_data(cwd)
                            if st.sidebar.button('Lets Make Prediction'):
                                test_generator = test_reading_data(cwd)
                                result = model_prediction(cwd,test_generator)
                                result_output_show(cwd,result,label_map)
            Reset_app = st.sidebar.checkbox('Click to Reset app')
            if Reset_app:
                st.error('It will delete all Labels and associated Data')
                del_label = st.selectbox('Are you Sure',('No','Yes'))
                if del_label == 'Yes':
                    st.error('Are you sure want to delete')
                    if st.button('Delete'):
                        try:
                            del_Lables_folders = []                    
                            for _, label_list,_  in os.walk(cwd+'/data/train/'):
                                for i in label_list:
                                    del_Lables_folders.append(label_list)
                            del_Lables_folders = del_Lables_folders[0]
                            for i in del_Lables_folders:
                                shutil.rmtree(os.path.join(cwd+'/data/train/'+str(i)))
                                shutil.rmtree(os.path.join(cwd+'/data/validation/'+str(i)))
                            shutil.rmtree(os.path.join(cwd+'/data/test/test/'))
                            os.remove(os.path.join(cwd+'/best.hdf5'))
                            st.text('done')
                        except Exception:
                            st.text('done')
    






