import shutil
import os
save_path = './evaluation_folder/eval'
files = os.listdir(save_path)
for i in range(10):
    os.mkdir(f'{save_path}/{i}')

num = 250
for file in files:
    img_ind = int(file.split('_')[0])
    if img_ind < num and img_ind >= 0:
        shutil.move(os.path.join(save_path, file), save_path + '/0')
    if img_ind < num*2 and img_ind > num:
        shutil.move(os.path.join(save_path, file), save_path + '/1')
    if img_ind < num*3 and img_ind >= num*2:
        shutil.move(os.path.join(save_path, file), save_path + '/2')
    if img_ind < num*4 and img_ind >= num*3:
        shutil.move(os.path.join(save_path, file), save_path + '/3')
    if img_ind < num*5 and img_ind >= num*4:
        shutil.move(os.path.join(save_path, file), save_path + '/4')
    if img_ind < num*6 and img_ind >= num*5:
        shutil.move(os.path.join(save_path, file), save_path + '/5')
    if img_ind < num*7 and img_ind >= num*6:
        shutil.move(os.path.join(save_path, file), save_path + '/6')
    if img_ind < num*8 and img_ind >= num*7:
        shutil.move(os.path.join(save_path, file), save_path + '/7')
    if img_ind < num*9 and img_ind >= num*8:
        shutil.move(os.path.join(save_path, file), save_path + '/8')
    if img_ind < num*10 and img_ind >= num*9:
        shutil.move(os.path.join(save_path, file), save_path + '/9')

