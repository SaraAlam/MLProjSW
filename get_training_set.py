import os
import PIL
import os.path
from PIL import Image
import numpy as np
import imageio
import albumentations as A

def get_seal_imgs_and_labels():
    # dir path
    yearly_file_names = ["Final_Training_Dataset_2020"]
    f = open("trainingNames.txt", "w")
    dir_path = os.curdir

    seal_idx = 0
    train_images, train_labels = list(), list()
    img_num = 0
    for yearly_file_name in yearly_file_names:
        yearly_file_path = os.path.join(dir_path, yearly_file_name)
        print("Total number of seals: " + str(len(os.listdir(yearly_file_path))))
        num_seals_to_train_on = 0

        for fname in os.listdir(yearly_file_path):
            seal_folder_path = os.path.join(yearly_file_path,fname)
            num_pics = len(os.listdir(seal_folder_path))
            if (num_pics>=2):
                f.write(fname+"\n")
                num_seals_to_train_on += 1
                img_num, train_images, train_labels = resize_and_add_to_X(img_num, seal_folder_path, train_images, train_labels, seal_idx)
                seal_idx+=1
        print("Number of seals with 2 or more pictures: " + str(num_seals_to_train_on))
    f.close()
    print(len(train_labels))
    train_images, train_labels = np.array(train_images), np.array(train_labels)
    print(train_images.shape)
    print(train_labels.shape)
    np.save("train_images", train_images)
    np.save("train_labels", train_labels)
    print("Total number of seals from both years in the training set: " + str(seal_idx))

def augmentation(seal_imgs, train_images, train_labels, seal_idx, count, dir_path2):
    for file in seal_imgs:
        f_img = os.path.join(dir_path2, str(file)+".jpeg")
        input_img = imageio.imread(f_img)
        c = 1
        toProduce = ((1/(count+1))*2)*8
        for i in [A.ShiftScaleRotate(p=1),
                A.augmentations.transforms.ChannelShuffle(p=1.0),
                A.RandomRotate90(p=1.0),
                A.VerticalFlip(p=1),
                A.Transpose(p=1.0),
                A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.5, rotate_limit=5, p=1.0),
                A.Blur(blur_limit=7, p =1.0),
                A.GridDistortion(p=1.0)]:
            if c > toProduce:
                break
            transform = i
            augmented_image = transform(image=input_img)['image']
            save = os.path.join(dir_path2, str(file)+"_"+str(c)+".jpeg")
            augmented_image = np.array(augmented_image)
            imageio.imwrite(save, augmented_image)
            train_images.append(augmented_image)
            train_labels.append(seal_idx)
            c += 1
        


    return train_images, train_labels

def resize_and_add_to_X(img_num, seal_folder_path, train_images, train_labels, seal_idx):
    seal_files = os.listdir(seal_folder_path)
    dir_path2 = os.path.join(os.getcwd(),"final_resized_imgs")
    if (not os.path.exists(dir_path2)):
        os.mkdir(dir_path2)
    seal_imgs = [img_num]
    count = 0
    non_img_files = []
    for file in seal_files:

            file_extention = ""
            if file.endswith("png"):
                file_extention = "png"
            if file.endswith("jpeg"):
                file_extention = "jpeg"

            if file_extention!="":
                count+=1
                train_labels.append(seal_idx)
                f_img = os.path.join(seal_folder_path, file)
                img = Image.open(f_img)
                img = img.resize((224,224))
                if file_extention=="png":
                    train_images.append(np.asarray(img)[:,:,:3])
                    f2 = os.path.join(dir_path2, str(img_num)+".png")
                else:
                    train_images.append(np.asarray(img))
                    f2 = os.path.join(dir_path2, str(img_num)+".jpeg")
                img.save(f2)
                img_num += 1
                seal_imgs.append(img_num)
            else:
                non_img_files.append(file)
    
    train_images, train_labels = augmentation(seal_imgs[:-1], train_images, train_labels, seal_idx, count, dir_path2)
    return img_num, train_images, train_labels



def get_seal_name(s):
    seal_name = ""
    for ch in s:
        if ch.isalpha():
            seal_name += ch
    return ch

def main():
    get_seal_imgs_and_labels()

if __name__=="__main__":
    main()
