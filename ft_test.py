import numpy as np
import sklearn.decomposition as dc
import matplotlib.pyplot as plt

left_arr = np.load("left.npy")
right_arr = np.load("right.npy")

# x = np.arange(0, 100)

# x = np.sin(x)

x_range = np.arange(0,len(left_arr)/30,1/30)
x_range = x_range[0:-1]
x_fft =  np.fft.fftshift(np.fft.fftfreq(len(left_arr),x_range[-1]/len(left_arr))) * 60
# ft = np.fft.fftshift(np.fft.fft(x))
# plt.plot(ft)
# plt.show()

ft1 = np.fft.fftshift(np.fft.fft(left_arr[:,0]-np.mean(left_arr[:,0])))
ft2 = np.fft.fftshift(np.fft.fft(left_arr[:,1]-np.mean(left_arr[:,1])))
ft3 = np.fft.fftshift(np.fft.fft(left_arr[:,2]-np.mean(left_arr[:,2])))
plt.plot(x_fft, np.power(np.abs(ft1),2),"b")
plt.plot(x_fft, np.power(np.abs(ft2),2),"g")
plt.plot(x_fft, np.power(np.abs(ft3),2),"r")

plt.figure()
plt.plot(x_range,left_arr[:,0], "b")
plt.plot(x_range,left_arr[:,1], "g")
plt.plot(x_range,left_arr[:,2], "r")
plt.show()


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
