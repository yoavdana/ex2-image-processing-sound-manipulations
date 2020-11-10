import numpy as np
import imageio as im
import matplotlib.pyplot as plt
import math as mt
import scipy as si
from scipy.io import wavfile as sci_io
import skimage.color as ski
##section 1
def DFT(signal):
    dft_mat=DFT_mat(len(signal))
    F_u=np.matmul(signal,dft_mat)
    return F_u
def DFT_mat(N):
    k, n = np.meshgrid(np.arange(N), np.arange(N))
    dft_mat=np.exp(-2*mt.pi*1j*k*n/N).astype('complex128')
    return dft_mat
def IDFT(fourier_signal):
    N=len(fourier_signal)
    idft_mat=IDFT_mat(N)
    signal=np.matmul(fourier_signal,idft_mat)/N
    return signal
def IDFT_mat(N):
    k, n = np.meshgrid(np.arange(N), np.arange(N))
    idft_mat=np.exp(2*mt.pi*1j*k*n/N).astype('complex128')
    return idft_mat
##section 1.2 2D DFT
def DFT2(image):
    N=image.shape[0]
    M=image.shape[1]
    x_dft=DFT_mat(N)
    y_dft=DFT_mat(M)
    F_u_v=x_dft.dot(image).dot(y_dft)
    return F_u_v
def IDFT2(fourier_image):
    N=fourier_image.shape[0]
    M=fourier_image.shape[1]
    x_idft=IDFT_mat(N)
    y_idft=IDFT_mat(M)
    f_x_y=x_idft.dot(fourier_image).dot(y_idft)/(N*M)
    return f_x_y
##section 2.1
def change_rate(filename, ratio):
    signal=sci_io.read(filename)[1]
    sample_rate=sci_io.read(filename)[0]
    new_sample_rate=int(ratio*sample_rate)
    sci_io.write('change_rate.wav',new_sample_rate,signal)##save new_signal

##section 2.2
def change_samples(filename, ratio):
    signal = sci_io.read(filename)[1]
    sample_rate = sci_io.read(filename)[0]
    new_data=resize(signal, ratio)
    sci_io.write('change_rate.wav',sample_rate,np.real(new_data))

    return new_data

def resize(data, ratio):
    F_data = np.fft.fft(data)
    F_data = np.fft.fftshift(F_data)
    if ratio>=1:
        rect = int(len(data) / ratio)
        new_F_s=F_data[int(len(F_data)/ 2) - int(rect / 2):int(len(F_data)/2) + int(rect/2)]

    else:
        pad_amount=int(len(F_data)/ratio)-len(F_data)
        if pad_amount%2==0:
            new_F_s=np.pad(F_data,(int(pad_amount/2),int(pad_amount/2)),
                           'constant')
        else:
            new_F_s = np.pad(F_data,(int(pad_amount / 2),int(pad_amount/2)+1),'constant')
    new_F_s=np.fft.ifftshift(new_F_s)
    new_signal = np.fft.ifft(new_F_s)
    return new_signal


def rect_filter(data, ratio):
    filter = np.zeros(len(data))
    rect = int(len(data) / ratio)
    filter[int(len(filter) / 2) - int(rect / 2):int(len(filter) / 2) + int(
        rect/2)] = 1
    return filter
data=sci_io.read('aria_4kHz.wav')[1]
new=np.real(change_samples('aria_4kHz.wav',0.5)).astype('int32')
#s=data-new
#r=np.max(np.abs(s))
#print(r)
f=np.arange(-int(len(data)/2),int(len(data)/2))
f1=np.arange(-int(len(new)/2),int(len(new)/2))
F_old=np.fft.fft(data)
F_old=np.fft.fftshift(F_old)
F_new=np.fft.fft(new)
F_new=np.fft.fftshift(F_new)
#new,old,new_f=resize(data, 2)
plt.plot(new)
plt.plot(data)
plt.show()
plt.plot(f,abs(F_old))
plt.plot(f1,abs(F_new))
plt.show()
print(data)
print(new)
#plt.show()

#signal=np.array([[1,0,1],[1,1,1],[1,1,1],[0,0,0]])
#toy=IDFT2(signal)
#real=np.fft.ifft2(signal)
