import numpy as np
import imageio
import cv2

def prctile_norm(x, min_prc=0, max_prc=100):
    # y = (x - np.percentile(x, min_prc)) / (np.percentile(x, max_prc) - np.percentile(x, min_prc) + 1e-7)
    y = x/ np.percentile(x, max_prc)
    y[y > 1] = 1
    y[y < 0] = 0
    return y


def cal_xs(onfo,xs,center1,center2):
    img = imageio.imread(onfo).astype(np.float)
    data = np.array(img[:, int(center1)])
    data = prctile_norm(data)
    real_xs=xs/data[int(center2)]
    return real_xs

def data_loader_focus(data_path, start, num,  patch_width,batch_size, dep,case,norm_flag=1):
    if num == 0:
        cell = np.zeros((1, batch_size))
    else:
        cell = np.random.choice(num, size=batch_size)
    depth = np.random.choice(30, size=batch_size)
    depth = depth + 35
    real_depth = (depth - 50) * 0.05
    image_batch = []
    label_batch = []
    if case==2:
        center1=192
        center2=104
        if dep==0.3:
            xs=[1,0.97]
        elif dep==0.5:
            xs=[1,0.91]
        elif dep==0.7:
            xs=[1,0.798]
    elif case==3:
        center1=128
        center2=69
        if dep==0.3:
            xs=[0.88,1,0.978]
        elif dep==0.5:
            xs=[0.739,1,0.916]
        elif dep==0.7:
            xs=[0.44,1,0.848]
    elif case==4:
        center1=96
        center2=52
        if dep==0.3:
            xs=[0.68,0.88,1,0.965]
        elif dep==0.5:
            xs=[0.428,0.732,1,0.91]
        elif dep==0.7:
            xs=[0.413,0.679,1,0.9]
    elif case==5:
        center1=64
        center2=34
        if dep==0.3:
            xs=[0.782,0.94,1,0.896,0.805]
        elif dep==0.5:
            xs=[0.465,0.87,1,0.772,0.518]
        elif dep==0.7:
            xs=[0.413,0.695,1,0.655,0.424]
    for i in range(0, batch_size):
        cur_img = []
        cur_img_on = []
        for j in range(0,case):
            path = data_path + 'cell' + str(int(cell[i]) + start) + '_' + str(depth[i]) +'_'+ str(j-int(case/2))+'.tif'
            # path1 = data_path + 'cell' + str(int(cell[i]) + start) + '_' + str(depth[i]) +'_'++'.tif'
            path_on = data_path + 'cell' + str(int(cell[i]) + start) + '_' + str(50) + '_'+ str(j-int(case/2))+'.tif'
            # print(path)
            a=cal_xs(path_on,xs[j],center1,center2)
            # print(a)
            img = imageio.imread(path).astype(np.float)
            if patch_width>1:
                img = img[:, center1-int(patch_width/2):center1+int(patch_width/2)]
                img = prctile_norm(img)
                img = img[center2-4:center2+4,:]*a
            else:
                img = img[:, center1]
                img = prctile_norm(img)
                img = img[center2-4:center2+4]*a
                img = img[:, np.newaxis]
            cur_img.append(img)

        if norm_flag:
            cur_img=np.array(cur_img)
            # img_on = prctile_norm(img_on)
        else:
            cur_img = cur_img / 65535
            # img_on = img_on / 65535
        label = np.zeros((1, 10))
        label[:, int((depth[i] - 1) / 20)] = 1
        image_batch.append(cur_img)
        label_batch.append(label)

    image_batch = np.array(image_batch)
    image_batch = np.transpose(image_batch, (0, 2, 3, 1))

    return image_batch, real_depth, cell
