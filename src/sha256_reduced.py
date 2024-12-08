import numpy as np

class SHA256:
    def __init__(self, rounds):
        self.rounds = rounds
        self.K = np.array([
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
            0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
            0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
            0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
            0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
        ], dtype=np.uint32)

        self.H = np.array([
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        ], dtype=np.uint32)

        np.seterr(over='ignore')

    def right_rotate(self, x, n):
        return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

    def sha256_compress(self, chunk):
        w = np.zeros(64, dtype=np.uint32)
        w[:16] = np.frombuffer(chunk, dtype=np.uint32).byteswap()

        for i in range(16, self.rounds):
            s0 = np.uint32(self.right_rotate(w[i-15], 7) ^ self.right_rotate(w[i-15], 18) ^ (w[i-15] >> 3))
            s1 = np.uint32(self.right_rotate(w[i-2], 17) ^ self.right_rotate(w[i-2], 19) ^ (w[i-2] >> 10))
            w[i] = (np.uint32(w[i-16]) + s0 + np.uint32(w[i-7]) + s1) & 0xFFFFFFFF

        a, b, c, d, e, f, g, h = self.H

        for i in range(self.rounds):
            S1 = np.uint32(self.right_rotate(e, 6) ^ self.right_rotate(e, 11) ^ self.right_rotate(e, 25))
            ch = (e & f) ^ (~e & g)
            temp1 = (np.uint32(h) + S1 + np.uint32(ch) + self.K[i] + w[i]) & 0xFFFFFFFF
            S0 = np.uint32(self.right_rotate(a, 2) ^ self.right_rotate(a, 13) ^ self.right_rotate(a, 22))
            maj = (a & b) ^ (a & c) ^ (b & c)
            temp2 = (S0 + np.uint32(maj)) & 0xFFFFFFFF

            h = g
            g = f
            f = e
            e = (d + temp1) & 0xFFFFFFFF
            d = c
            c = b
            b = a
            a = (temp1 + temp2) & 0xFFFFFFFF

        self.H = (self.H + [a, b, c, d, e, f, g, h]) & 0xFFFFFFFF

    def pad_message(self, message):
        message = bytearray(message)
        original_length = len(message) * 8
        message.append(0x80)

        while (len(message) * 8) % 512 != 448:
            message.append(0)

        message += original_length.to_bytes(8, 'big')
        return message

    def calculate_hash(self, message):
        message = self.pad_message(message)

        for i in range(0, len(message), 64):
            chunk = message[i:i+64]
            self.sha256_compress(chunk)

        return ''.join(f'{value:08x} ' for value in self.H)

if __name__ == "__main__":
    sha = SHA256(16)
    message = b"SHA-256!"
    hash_result = sha.calculate_hash(message)
    print("SHA-256 Hash:", hash_result)
