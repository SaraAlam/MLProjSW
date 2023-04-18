import os
import PIL
import os.path
from PIL import Image
import numpy as np

def get_seal_folders():
    # dir path
    yearly_file_names = ["Final_Training_Dataset_2019", "Final_Training_Dataset_2020"]
    f = open("trainingNames.txt", "w")
    dir_path = r'C:\Users\sarah\Downloads\ApplMLlabs\AppMLProjSpr23SaraWael'

    seal_idx = 0
    train_images, train_labels = list(), list()

    for yearly_file_name in yearly_file_names:
        yearly_file_path = os.path.join(dir_path, yearly_file_name)
        print("Total number of seals: " + str(len(os.listdir(yearly_file_path))))
        num_seals_to_train_on = 0

        for fname in os.listdir(yearly_file_path):
            seal_name = get_seal_name(fname)
            seal_folder_path = os.path.join(yearly_file_path,fname)
            num_pics = len(os.listdir(seal_folder_path))
            if (num_pics>=2):
                f.write(fname+"\n")
                num_seals_to_train_on += 1
                train_images, train_labels = resize_and_add_to_X(seal_name, seal_folder_path, train_images, train_labels, seal_idx)
                seal_idx+=1
        print("Number of seals with 2 or more pictures: " + str(num_seals_to_train_on))
    f.close()
    train_images, train_labels = np.array(train_images), np.array(train_labels)
    np.save("train_images", train_images)
    np.save("train_labels", train_labels)
    print("Total number of seals from both years in the training set: " + str(seal_idx))

def resize_and_add_to_X(seal_name, seal_folder_path, train_images, train_labels, seal_idx):
    seal_files = os.listdir(seal_folder_path)
    dir_path2 = os.path.join(os.getcwd(),"final_resized_imgs")
    img_num = len(train_labels)
    for file in seal_files:
            train_labels.append(seal_idx)

            file_extention = ""
            if file.endswith("png"):
                file_extention = "png"
            if file.endswith("jpeg"):
                file_extention = "jpeg"

            if file_extention!="":
                f_img = os.path.join(seal_folder_path, file)
                img = Image.open(f_img)
                img = img.resize((224,224))
                if file_extention=="png":
                    train_images.append(np.asarray(img)[:,:,:3])
                else:
                    train_images.append(np.asarray(img))
                f2 = os.path.join(dir_path2, str(img_num)+".png")
                img.save(f2)
                img_num += 1
    return train_images, train_labels

def get_seal_name(s):
    seal_name = ""
    for ch in s:
        if ch.isalpha():
            seal_name += ch
    return ch

def main():
    get_seal_folders()

if __name__=="__main__":
    main()