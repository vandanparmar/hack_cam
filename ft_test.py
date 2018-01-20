import numpy as np
import sklearn.decomposition as dc
import matplotlib.pyplot as plt
from matplotlib import colors,cm
import json


def find_nearest(array,value):
	return (np.abs(array-value)).argmin()

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma
 

left_arr = np.load("left_2035.npy")
right_arr = np.load("right_2035.npy")


frames = 300
step = 10
fps = 10
sample = left_arr
length = frames
heart_rates = []
print(len(sample))
print(len(sample)/step)

def get_hr_seq(frames,step,fps,sample):
	fts = []
	heart_rates = []
	heart_rates_int = []
	for i in range(0,int((len(sample)-frames)/step)):
		part = sample[i*step:i*step+frames,1]
		print(part.shape)
		ft = np.fft.fftshift(np.fft.fft(part))
		ft = np.power(np.abs(ft),2)
		x_range = np.arange(0,frames/fps,1/fps)
		x_range = x_range[0:-1]
		x_fft =  np.fft.fftshift(np.fft.fftfreq(frames,x_range[-1]/frames)) * 60
		pos_50 = find_nearest(x_fft,50)
		pos_150 = find_nearest(x_fft,150)
		print(pos_50,pos_150)
		ft = ft[pos_50:pos_150]
		heart_rate_int = np.multiply(x_fft[pos_50:pos_150],ft).sum()/ft.sum()
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

plt.plot(heart_rates_right,'r',label='right')
plt.plot(heart_rates_left,'b',label='left')
plt.plot(heart_rates_int_right,'g',label='int right')
plt.plot(heart_rates_int_left,'k',label = 'int left')
plt.legend()
plt.show()

to_save = (np.array(heart_rates_left) + np.array(heart_rates_right)) /2
to_save = {'hr':to_save.tolist()}


with open('data.json', 'w') as outfile:
    json.dump(to_save, outfile)

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
