import math

class _BlumBlumShub:
    def __init__(self, seed: int):
        self.p = 23
        self.q = 19
        
        self.M = self.p * self.q
        
        while math.gcd(seed, self.M) != 1:
            seed += 1

        self.x = (seed * seed) % self.M

    def next_int(self) -> int:
        self.x = (self.x * self.x) % self.M
        return self.x