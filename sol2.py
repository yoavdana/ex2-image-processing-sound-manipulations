import numpy as np
import imageio as im
import scipy as si
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
from scipy.io import wavfile as sci_io
from skimage.color import rgb2gray
GRAY_SCALE=2
RGB=3
MAX_PIXEL=255

##section 1
def DFT(signal):
    ##calculate the fourier transform for 1D signal
    T_signal=signal.reshape(len(signal),)
    dft_mat=DFT_mat(len(T_signal))
    F_u=T_signal.dot(dft_mat)
    F_u = np.reshape(F_u, signal.shape)
    return F_u
def DFT_mat(N):
    ##generate the DFT matrix for 1D signal
    k, n = np.meshgrid(np.arange(N), np.arange(N))
    dft_mat=np.exp(-2*np.pi*1j*k*n/N).astype('complex128')
    return dft_mat
def IDFT(fourier_signal):
    ##calculate the inverse fourier transform for 1D signal
    N=len(fourier_signal)
    T_signal = fourier_signal.reshape(N, )
    idft_mat=IDFT_mat(N)
    signal= T_signal.dot(idft_mat)/N
    signal = np.reshape(signal, fourier_signal.shape)
    return signal
def IDFT_mat(N):
    ##generate the IDFT matrix for 1D signal
    k, n = np.meshgrid(np.arange(N), np.arange(N))
    idft_mat=np.exp(2*np.pi*1j*k*n/N).astype('complex128')
    return idft_mat
##section 1.2 2D DFT
def DFT2(image):
    N=image.shape[0]#row size
    M=image.shape[1]#columb size
    T_image = image.reshape(N,M)
    x_dft=DFT_mat(N)#1D DFT for x
    y_dft=DFT_mat(M)#1D DFT for y
    F_u_v=x_dft.dot(T_image).dot(y_dft)#2d transfor by mat multiplaction
    F_u_v=np.reshape(F_u_v, image.shape)
    return F_u_v
def IDFT2(fourier_image):
    N=fourier_image.shape[0]
    M=fourier_image.shape[1]
    T_image = fourier_image.reshape(N, M)
    x_idft=IDFT_mat(N)
    y_idft=IDFT_mat(M)
    f_x_y=x_idft.dot(T_image).dot(y_idft)/(N*M)
    f_x_y= np.reshape(f_x_y, fourier_image.shape)
    return f_x_y
##section 2.1
def change_rate(filename, ratio):
    #the function change a given sample rate for a given wav file
    signal=sci_io.read(filename)[1]
    sample_rate=sci_io.read(filename)[0]
    new_sample_rate=int(ratio*sample_rate)
    sci_io.write('change_rate.wav',new_sample_rate,signal)##save new_signal

##section 2.2
def change_samples(filename, ratio):
    #the function change the speed of a signal by filtering the frequancy space
    sample_rate,signal = sci_io.read(filename)
    new_data=resize(signal, ratio)
    music_data=np.real(new_data)
    sci_io.write('change_samples.wav.',sample_rate,music_data)
    return np.real(new_data).astype('float64')

def resize(data, ratio):
    F_data=DFT(data)
    F_data = np.fft.fftshift(F_data)#shift the fft to create simetry for the
    # filter
    if ratio>=1:#in case of speeding up
        rect = int(len(data)/ratio)#filter size
        left_bound=int(len(F_data)/ 2 -(rect / 2))#set the bounds of the filter
        right_bound=int(len(F_data)/ 2 +(rect / 2))
        new_F_s=F_data[left_bound:right_bound]#slice the frquancy space(
        # filterring)
    else:#in case of slowing down
        pad_amount=int(len(F_data)/ratio)-len(F_data) #amount of zeros to be
        # added
        if pad_amount%2==0:#in case #of zeros is even
            new_F_s=np.pad(F_data.reshape(len(F_data),),(int(pad_amount/2),int(pad_amount/2)),'constant')
        else:#in case #of zeros is odd
            new_F_s = np.pad(F_data.reshape(len(F_data),),(int(pad_amount / 2),int(pad_amount/2)+1),'constant')
    new_F_s=np.fft.ifftshift(new_F_s)
    new_signal = IDFT(new_F_s)
    return new_signal

def resize_spectrogram(data,ratio):
    stft_mat = stft(data)
    new_stft_mat = np.zeros(((len(stft_mat[:, 1])), int(len(stft_mat[1,:])/ratio)))
    for i in range(len(stft_mat[:,1])):
        new_stft_mat[i,:]=resize(stft_mat[i,:],ratio)
    new_data=istft(new_stft_mat)
    return new_data
def resize_vocoder(data, ratio):
    stft_mat=stft(data)
    stft_vecode=phase_vocoder(stft_mat,ratio)
    new_data=istft(stft_vecode)
    return new_data

##section 3.1
def conv_der(im):
    dx_op=np.array([0.5,0,-0.5]).reshape((1,3))
    dy_op=np.array([0.5,0,-0.5]).reshape((3,1))
    dx=si.signal.convolve2d(im,dx_op,'same')
    dy=si.signal.convolve2d(im, dy_op, 'same')
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude
##section 3.2
def fourier_der(im):
    im_dft=DFT2(im)
    im_dft_shift=np.fft.fftshift(im_dft)
    u_len=im.shape[0]#u axes len
    v_len = im.shape[1]#v axes len
    if v_len % 2==0:
        v_axes=np.arange(-int(v_len/2),int(v_len/2))*2j*np.pi/v_len
    else:
        v_axes = np.arange(-int(v_len / 2),int(v_len / 2)+1) * 2j * np.pi / v_len
    Fourier_der_v=im_dft_shift.reshape(u_len,v_len)*v_axes
    dy=IDFT2(Fourier_der_v)# div o y
    if u_len % 2 == 0:
        u_axes = np.arange(-int(u_len / 2),int(u_len / 2)) * 2j * np.pi / u_len
    else:
        u_axes = np.arange(-int(u_len / 2), int(u_len / 2) + 1) * 2j * np.pi / u_len
    Fourier_der_u=np.transpose(np.transpose(im_dft_shift.reshape(u_len,v_len))*u_axes)
    dx=IDFT2(Fourier_der_u)#div of x
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    magnitude = np.reshape( magnitude, im.shape)
    return magnitude
##ex2 helper
def read_image(filename, representation):
    #the function will read an image file and return a normalizes array of
    # its intesitys
    image=im.imread(filename).astype(np.float64)
    if np.amax(image)>1:
        image=image.astype(np.float64)/MAX_PIXEL
    if representation==2 and image.ndim!=GRAY_SCALE:#return RGB from RGB file
        return image
    elif representation==1 and image.ndim==RGB:#return grayscale from RGB file
        return rgb2gray(image)
    elif representation==1 and image.ndim==GRAY_SCALE: #return grayscale from
        # grayscale file
        return image

def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    time_steps = np.arange(spec.shape[1]) * ratio
    time_steps = time_steps[time_steps < spec.shape[1]]

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase
    return warped_spec
