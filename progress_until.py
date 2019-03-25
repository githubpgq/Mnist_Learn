import sys
import time


class Progress:
    def __init__(self, total, name):
        self.total = total
        self.name = name
        self.now = 0.0
        self.step = 0.0001
        self.start_time = time.time()

    def updata(self, now):
        if (now >= self.total):
            sys.stdout.write(self.name + ": %.2f%%   \r" % (
            (now / self.total) * 100))
            finish_time = time.time() - self.start_time
            print("")
            print("完成！ 用时: %.4f s  " % (finish_time))
            return
        if (now <= self.now):
            return
        sys.stdout.write(self.name + ": %.2f%%   \r" % (
            (now / self.total) * 100))
        sys.stdout.flush()
        next_now = self.now + self.step * self.total
        self.now = next_now if (next_now > now) else now
