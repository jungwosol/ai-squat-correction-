from klib2_python import *
import numpy as np

sensor = KLib("127.0.0.1", 3800)

sensor.start()

sensor.read()

arr = np.array(sensor.dataMatrix)

print(arr.reshape((32,10)))