import numpy as np
from numba import njit

class Utils:
    h0 = 0x6A09E667
    h1 = 0xBB67AE85
    h2 = 0x3C6EF372
    h3 = 0xA54FF53A
    h4 = 0x510E527F
    h5 = 0x9B05688C
    h6 = 0x1F83D9AB
    h7 = 0x5BE0CD19

    k = np.array([
        0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
        0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
        0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
        0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
        0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
        0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
        0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
        0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
    ], dtype=np.uint32)

    @staticmethod
    def right_rotate(value, bits):
        return ((value >> bits) | (value << (32 - bits))) & 0xFFFFFFFF

    @staticmethod
    def getS0(value):
        return Utils.right_rotate(value, 7) ^ Utils.right_rotate(value, 18) ^ (value >> 3)

    @staticmethod
    def right_rotate1(value, bits):
        return np.uint32((value >> bits) | (value << (32 - bits)))

    @staticmethod
    def getS01(value):
        rotated1 = np.uint32(Utils.right_rotate1(value, 7))
        rotated2 = np.uint32(Utils.right_rotate1(value, 18))
        shifted = np.uint32(value >> 3)
        return rotated1 ^ rotated2 ^ shifted

    @staticmethod
    @njit
    def right_rotate3(value, bits):
        return ((value >> bits) | (value << (32 - bits))) & 0xFFFFFFFF

    @staticmethod
    @njit
    def getS03(value):
        rotated1 = Utils.right_rotate3(value, 7)
        rotated2 = Utils.right_rotate3(value, 18)
        shifted = value >> 3
        return rotated1 ^ rotated2 ^ shifted

    @staticmethod
    def getS1(value):
        return Utils.right_rotate(value, 17) ^ Utils.right_rotate(value, 19) ^ (value >> 10)

    @staticmethod
    def getΣ0(value):
        return Utils.right_rotate(value, 2) ^ Utils.right_rotate(value, 13) ^ Utils.right_rotate(value, 22)

    @staticmethod
    def getDeltaΣ0(value, delta):
        return Utils.getΣ0((value + delta) & 0xFFFFFFFF) - Utils.getΣ0(value)

    @staticmethod
    def getΣ1(value):
        return Utils.right_rotate(value, 6) ^ Utils.right_rotate(value, 11) ^ Utils.right_rotate(value, 25)

    @staticmethod
    def getDeltaΣ1(value, delta):
        return Utils.getΣ1((value + delta) & 0xFFFFFFFF) - Utils.getΣ1(value)

    @staticmethod
    def getMa(a, b, c):
        return (a & b) ^ (a & c) ^ (b & c)

    @staticmethod
    def getDeltaMa(a, b, c, delta1, delta2, delta3):
        return Utils.getMa(a + delta1, b + delta2, c + delta3) - Utils.getMa(a, b, c)

    @staticmethod
    def getCh(e, f, g):
        return (e & f) ^ (~e & g)

    @staticmethod
    def getDeltaCh(e, f, g, delta1, delta2, delta3):
        return Utils.getCh(e + delta1, f + delta2, g + delta3) - Utils.getCh(e, f, g)

    @staticmethod
    def getT2(a, b, c):
        return Utils.getΣ0(a) + Utils.getMa(a, b, c)
