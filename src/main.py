
import numpy as np

from src.utils import Utils
np.seterr(over='ignore')


class CalculateCollisions:

    def __init__(self):
        self.alpha = 1812291026
        #self.beta = np.zeros(6, dtype =  np.uint32)
        self.beta = np.array([0, 0, 0xb806d01d, 0xe00fc62f, 0x0002441e, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint32)
        self.words = np.array([0xbba73b74, 0xaa8d8059, 0x0e4d263c, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint32)
        self.t1 = np.zeros(11, dtype =  np.uint32)
        self.t2 = np.zeros(11, dtype =  np.uint32)
        self.h0 = np.array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19], dtype=np.uint32)
        self.a = np.array([self.h0[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint32)
        self.b = np.array([self.h0[1], self.h0[0], 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint32)
        self.c = np.array([self.h0[2], self.h0[1], self.h0[0], 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint32)
        self.d = np.array([self.h0[3], self.h0[2], self.h0[1], self.h0[0], 0, 0, 0, 0, 0, 0, 0], dtype=np.uint32)
        self.e = np.array([self.h0[4], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint32)
        self.f = np.array([self.h0[5], self.h0[4], 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint32)
        self.g = np.array([self.h0[6], self.h0[5], self.h0[4], 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint32)
        self.h = np.array([self.h0[7], self.h0[6], self.h0[5], self.h0[4], 0, 0, 0, 0, 0, 0, 0], dtype=np.uint32)

    def calc_next_ae(self, i):
        self.t1[i] = (self.h[i] + Utils.getΣ1(self.e[i]) + Utils.getCh(self.e[i], self.f[i], self.g[i]) + Utils.k[i] +
                      self.words[i]) & 0xFFFFFFFF
        self.t2[i] = (Utils.getΣ0(self.a[i]) + Utils.getMa(self.a[i], self.b[i], self.c[i])) & 0xFFFFFFFF
        self.a[i + 1] = (self.t1[i] + self.t2[i]) & 0xFFFFFFFF
        self.e[i + 1] = (self.d[i] + self.t1[i]) & 0xFFFFFFFF

    def calcNextBCDEFG(self, i):
        self. b[i + 1] = self.a[i]
        self.c[i + 1] = self.b[i]
        self.d[i + 1] = self.c[i]
        self.f[i + 1] = self.e[i]
        self.g[i + 1] = self.f[i]
        self.h[i + 1] = self.g[i]

    def step1(self):
        print("Start processing step 1 :")
        max = 2 ** 32
        counter = 0
        #self.calcNextBCDEFG(0)
        while True:
            #self.words[0] = (np.random.randint(-2**32, 2**32, dtype=np.int64))
            counter += 1
            #self.alpha = (np.random.randint(-2**32, 2**32, dtype=np.int64))
            self.calc_next_ae(0)
            beta2_1 = ((self.alpha - Utils.getDeltaΣ0(self.a[1], self.alpha) -
                       Utils.getDeltaMa(self.a[1], self.b[1], self.c[1], self.alpha, 0, 0))) & 0xFFFFFFFF
            beta2_2 = ((Utils.getDeltaΣ1(self.e[1], self.alpha) +
                   Utils.getDeltaCh(self.e[1], self.f[1], self.g[1], self.alpha, 0, 0))) & 0xFFFFFFFF
            #print(hex(beta2_1), hex(beta2_2), beta2_1 - beta2_2)
            if (abs(beta2_1 - beta2_2) < max):
                max = abs(beta2_1 - beta2_2)
                print(counter,"=",max)
            if beta2_1 == beta2_2:
                self.beta[2] = beta2_1
                break

    def step2(self):
        print("Start processing step 2 :")
        max = 2 ** 32
        counter = 0
        self.calcNextBCDEFG(1)
        while True:
            #self.words[1] = (np.random.randint(-2**32, 2**32, dtype=np.int64))
            counter += 1
            self.calc_next_ae(1)
            beta3_1 = ((- Utils.getDeltaΣ0(self.a[2], self.alpha) -
                                 Utils.getDeltaMa(self.a[2], self.b[2], self.c[2], self.alpha, self.alpha, 0))) & 0xFFFFFFFF
            beta3_2 = ((Utils.getDeltaΣ1(self.e[2], self.beta[2]) +
                                 Utils.getDeltaCh(self.e[2], self.f[2], self.g[2], self.beta[2], self.alpha, 0))) & 0xFFFFFFFF
            #print(hex(beta3_1), hex(beta3_2), beta3_1 - beta3_2)
            if (abs(beta3_1 - beta3_2) < max):
                max = abs(beta3_1 - beta3_2)
                print(counter,"=",max)
            if beta3_1 == beta3_2:
                self.beta[3] = beta3_1
                break

    def step3(self):
        print("Start processing step 3 :")
        max = 2 ** 32
        counter = 0
        self.calcNextBCDEFG(2)
        while True:
            #self.words[2] = (np.random.randint(-2**32, 2**32, dtype=np.int64))
            counter += 1
            self.calc_next_ae(2)
            beta4_1 = (-Utils.getDeltaMa(self.a[3], self.b[3], self.c[3],0, self.alpha, self.alpha)) & 0xFFFFFFFF
            beta4_2 = ((Utils.getDeltaΣ1(self.e[3], self.beta[3]) +
                        Utils.getDeltaCh(self.e[3], self.f[3], self.g[3], self.beta[3], self.beta[2], self.alpha))) & 0xFFFFFFFF
            #print(hex(beta3_1), hex(beta3_2), beta3_1 - beta3_2)
            if (abs(beta4_1 - beta4_2) < max):
                max = abs(beta4_1 - beta4_2)
                print(counter,"=",max)
            if beta4_1 == beta4_2:
                self.beta[4] = beta4_1
                break

    def step4(self):
        print("Start processing step 4 :")
        max = 2 ** 32
        counter = 0
        self.calcNextBCDEFG(3)
        while True:
            self.words[3] = (np.random.randint(-2**32, 2**32, dtype=np.int64))
            counter += 1
            self.calc_next_ae(3)
            beta5_1 = (-Utils.getDeltaMa(self.a[4], self.b[4], self.c[4],0, 0, self.alpha) + self.alpha) & 0xFFFFFFFF
            beta5_2 = ((Utils.getDeltaΣ1(self.e[4], self.beta[4]) +
                        Utils.getDeltaCh(self.e[4], self.f[4], self.g[4], self.beta[4], self.beta[3], self.beta[2])) + 2*self.alpha) & 0xFFFFFFFF
            #print(hex(beta3_1), hex(beta3_2), beta3_1 - beta3_2)
            if (abs(beta5_1 - beta5_2) < max):
                max = abs(beta5_1 - beta5_2)
                print(counter,"=",max)
            if beta5_1 == beta5_2:
                self.beta[4] = beta5_1
                break

    def print(self, array):
        hex_array = [f'{x:08x}' for x in array]
        print(hex_array)

calculations = CalculateCollisions()
calculations.step1()
calculations.print(calculations.words)
calculations.print(calculations.beta)
print(hex(calculations.alpha))
calculations.step2()
calculations.print(calculations.words)
calculations.print(calculations.beta)
print(hex(calculations.alpha))
calculations.step3()
calculations.print(calculations.words)
calculations.print(calculations.beta)
print(hex(calculations.alpha))
calculations.step4()
calculations.print(calculations.words)
calculations.print(calculations.beta)
print(hex(calculations.alpha))

