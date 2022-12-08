import numpy as np
from PIL import Image as PILImage
from PIL import Image


def draw_cicle(shape, diamiter):

    assert len(shape) == 2
    TF = np.zeros(shape, dtype="bool")
    center = np.array(TF.shape) / 2.0

    for iy in range(shape[0]):
        for ix in range(shape[1]):
            TF[iy, ix] = (iy - center[0]) ** 2 + (ix - center[1]) ** 2 < diamiter ** 2
    return TF


def filter_circle(TFcircle, fft_img_channel):
    temp = np.zeros(fft_img_channel.shape[:2], dtype=complex)
    temp[TFcircle] = fft_img_channel[TFcircle]
    return temp


def inv_FFT_all_channel(fft_img):
    img_reco = []
    for ichannel in range(fft_img.shape[2]):
        img_reco.append(np.fft.ifft2(np.fft.ifftshift(fft_img[:, :, ichannel])))
    img_reco = np.array(img_reco)
    img_reco = np.transpose(img_reco, (1, 2, 0))
    return img_reco



def high_pass_filter(x, severity):
    x = x.astype("float32") / 255.
    c = [.01, .02, .03, .04, .05][severity - 1]

    d = int(c * x.shape[0])
    TFcircle = draw_cicle(shape=x.shape[:2], diamiter=d)
    TFcircle = ~TFcircle

    fft_img = np.zeros_like(x, dtype=complex)
    for ichannel in range(fft_img.shape[2]):
        fft_img[:, :, ichannel] = np.fft.fftshift(np.fft.fft2(x[:, :, ichannel]))

    # For each channel, pass filter
    fft_img_filtered = []
    for ichannel in range(fft_img.shape[2]):
        fft_img_channel = fft_img[:, :, ichannel]
        temp = filter_circle(TFcircle, fft_img_channel)
        fft_img_filtered.append(temp)
    fft_img_filtered = np.array(fft_img_filtered)
    fft_img_filtered = np.transpose(fft_img_filtered, (1, 2, 0))
    x = np.clip(np.abs(inv_FFT_all_channel(fft_img_filtered)), a_min=0, a_max=1)

    x = PILImage.fromarray((x * 255.).astype("uint8"))
    return x

def resize_image(img, target_dim):
    new_img = img.resize(target_dim, Image.ANTIALIAS)
    return new_img


if __name__ == "__main__":
    # data_dir="/data/DataSets/"
    # root_path = os.path.join(os.path.dirname(__file__), data_dir, 'PACS/pacs_label')
    # label_file = os.path.join(root_path, "photo_train_kfold.txt") #photo/dog/056_0027.jpg 1
 #   print(label_file)

   # label_file = '/data/DataSets/PACS/pacs_label/photo_train_kfold.txt'
   # img_dim = (224, 224)

    # imagedata = []
    # with open(label_file, "r") as f_label:
    #     for line in f_label:
    #         temp = line[:-1].split(" ")
    #       #  print(temp)  ['photo/person/253_0426.jpg', '7']
    #         img = Image.open(os.path.join('/data/DataSets/PACS/kfold', temp[0]))
    #         img = resize_image(img, img_dim)
    #       #  print(type(img)) <class 'PIL.Image.Image'> np.array和copy都可转为 <class 'numpy.ndarray'>
    #         imagedata.append(np.array(img))
    # imagedata = np.array(imagedata)  #list-> <class 'numpy.ndarray'>
    # x = imagedata[0]  #如果取0的话，list和array的结果一样
    # im = Image.fromarray(x)
    # im.save('/data/wangna/GFNet1/0.jpg')
    # x_ = np.copy(x)
    # x_ = high_pass_filter(x_, severity=1)
    # x_.save('/data/wangna/GFNet1/1.jpg')

#-------------------------------------------------------------------------------------------
    #水狗 /data/wangna/dataset/PACS/kfold/photo/dog/n02103406_6530.jpg
    #草狗 /056_0079.jpg  n02103406_5216.jpg   056_0065.jpg
    #大象纹理 /data/wangna/dataset/PACS/kfold/photo/elephant/064_0011.jpg  064_0041
    img = Image.open('/data/wangna/dataset/PACS/kfold/photo/horse/n02374451_2743.jpg')
    img = resize_image(img, (224, 224))
    img = np.array(img)
    im = Image.fromarray(img)
    im.save('/data/wangna/GFNet1/GFNet-master/imageSUPP/15.jpg')
    x_ = np.copy(img)
    x_ = high_pass_filter(x_, severity=2)
    x_.save('/data/wangna/GFNet1/GFNet-master/imageSUPP/152.jpg')

    HPimg = Image.open('/data/wangna/GFNet1/GFNet-master/imageSUPP/152.jpg')
    img1_fft = np.fft.fft2(HPimg, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
 #   img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
  #  img1_abs_ = np.copy(img1_abs)
  #  img1_pha_ = np.copy(img1_pha)
  #   b = np.ones((224,224,3), dtype=np.float64)
  #   b = b*60000
    img1_abs_ = img1_abs * 1.0    #振幅 0.6
    img1_pha_ = img1_pha * 1.3  #相位 0.6
  #  img1_abs_ = np.fft.ifftshift(img1_abs_, axes=(0, 1))
    img21 = img1_abs_ * (np.e ** (1j * img1_pha_))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    im21 = Image.fromarray(img21)
    im21.save('/data/wangna/GFNet1/GFNet-master/imageSUPP/scale152.jpg')