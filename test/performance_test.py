import time
import numpy as np
from numba import njit


from src.utils import Utils

@njit
def right_rotate3(value, bits):
    return ((value >> bits) | (value << (32 - bits))) & 0xFFFFFFFF

@njit
def getS03(value):
    rotated1 = right_rotate3(value, 7)
    rotated2 = right_rotate3(value, 18)
    shifted = value >> 3
    return rotated1 ^ rotated2 ^ shifted

start_time = time.time()
initial_value = np.random.randint(-2**32, 2**32, dtype=np.int64) & 0xFFFFFFFF;
value = initial_value
while True:
    value = Utils.getS0(value)
    if value == initial_value:
        break
end_time = time.time()
print(hex(value))
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")



start_time = time.time()
value = initial_value
while True:
    value = getS03(value)
    if value == initial_value:
        break
end_time = time.time()
print(hex(value))
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")


start_time = time.time()
value = initial_value
while True:
    value = Utils.getS01(value)
    if value == initial_value:
        break
end_time = time.time()
print(hex(value))
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
