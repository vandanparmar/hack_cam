import numpy as np
import sklearn.decomposition as dc
import matplotlib.pyplot as plt
from matplotlib import colors,cm

def find_nearest(array,value):
	return (np.abs(array-value)).argmin()

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def nan_remover(array):
	new_arr = []
	for index, elem in enumerate(array):
		print(elem)
		if str(elem) != 'nan':
			new_arr.append(elem)
		else:
			new_arr.append(new_arr[-1])
	return np.array(new_arr)


left_arr = np.load("left_sj.npy")
right_arr = np.load("right_sj.npy")


frames = 100
step = 24
fps = 24
heart_rates = []

def get_hr_seq(frames,step,fps,sample):
	fts = []
	heart_rates = []
	heart_rates_int = []
	for i in range(0,int((len(sample)-frames)/step)):
		part = nan_remover(sample[i*step:i*step+frames,1])
		ft = np.fft.fftshift(np.fft.fft(part))
		ft = np.power(np.abs(ft),2)
		x_range = np.arange(0,frames/fps,1/fps)
		x_range = x_range[0:-1]
		x_fft =  np.fft.fftshift(np.fft.fftfreq(frames,x_range[-1]/frames)) * 60
		pos_50 = find_nearest(x_fft,50)
		pos_200 = find_nearest(x_fft,200)
		print(pos_50,pos_200)
		ft = ft[pos_50:pos_200]
		heart_rate_int = np.multiply(x_fft[pos_50:pos_200],ft).sum()/ft.sum()
		if str(heart_rate_int) == 'nan':
			print(ft)
			print(part)
			plt.plot(x_fft[pos_50:pos_200], ft)
			plt.show()
		heart_rates_int.append(heart_rate_int)
		fts.append(ft)
		heart_rates.append(x_fft[ft.argmax()+pos_50])
	return heart_rates,heart_rates_int,fts

heart_rates_left,heart_rates_int_left,fts_left = get_hr_seq(frames,step,fps,left_arr)
heart_rates_right,heart_rates_int_right,fts_right = get_hr_seq(frames,step,fps,right_arr)

# fts_left = np.array(fts_left)


# my_cmap = cm.get_cmap('BuPu')
# my_cmap.set_bad((0.9686275,0.9882359411,0.9921568627))
# plt.pcolor(np.absolute(fts_left), norm=colors.LogNorm(), cmap=my_cmap)
# plt.colorbar()
# plt.show()

# print(fts_left.shape)
# plt.imshow(fts_left)
# plt.show()

# heart_rates_left = movingaverage(heart_rates_left,20)

# plt.plot(heart_rates_right,'r',label='right')
# plt.plot(heart_rates_left,'b',label='left')
plt.plot(heart_rates_int_right,'g',label='int right')
plt.plot(heart_rates_int_left,'k',label = 'int left')
plt.plot((np.array(heart_rates_int_right) + np.array(heart_rates_int_left))/2, 'c', label = 'avg')
plt.legend()
plt.show()

# x = np.arange(0, 100)

# x = np.sin(x)

# x_range = np.arange(0,len(left_arr)/30,1/30)
# x_range = x_range[0:-1]
# x_fft =  np.fft.fftshift(np.fft.fftfreq(len(left_arr),x_range[-1]/len(left_arr))) * 60
# # ft = np.fft.fftshift(np.fft.fft(x))
# # plt.plot(ft)
# # plt.show()

# ft1 = np.fft.fftshift(np.fft.fft(left_arr[:,0]-np.mean(left_arr[:,0])))
# ft2 = np.fft.fftshift(np.fft.fft(left_arr[:,1]-np.mean(left_arr[:,1])))
# ft3 = np.fft.fftshift(np.fft.fft(left_arr[:,2]-np.mean(left_arr[:,2])))
# plt.xlim([50,150])
# plt.plot(x_fft, np.power(np.abs(ft1),2),"b")
# plt.plot(x_fft, np.power(np.abs(ft2),2),"g")
# plt.plot(x_fft, np.power(np.abs(ft3),2),"r")

# plt.figure()
# plt.plot(x_range,left_arr[:,0], "b")
# plt.plot(x_range,left_arr[:,1], "g")
# plt.plot(x_range,left_arr[:,2], "r")
# plt.show()


# import numpy as np
# import sklearn.decomposition as dc
# import matplotlib.pyplot as plt

# left_arr = np.load("left.npy")
# right_arr = np.load("right.npy")

# # x = np.arange(0, 100)

# # x = np.sin(x)
# x_range = np.arange(0,len(left_arr)/30,1/30)
# x_range = x_range[0:-1]
# left_arr = np.sin(x_range)
# x_fft =  np.fft.fftshift(np.fft.fftfreq(len(left_arr),x_range[-1]/len(left_arr)))
# # ft = np.fft.fftshift(np.fft.fft(x))
# # plt.plot(ft)
# # plt.show()

# ft3 = np.fft.fftshift(np.fft.fft(left_arr-np.mean(left_arr)))
# plt.plot(x_fft, np.power(np.abs(ft3),2),"r")

# plt.figure()
# plt.plot(x_range,left_arr, "b")
# plt.show()
