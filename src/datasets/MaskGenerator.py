import numpy as np
import cv2

class MaskGenerator(object):
    def __init__(self, height, width, channels=3, num_irregulars=20):
        self.height         = height
        self.width          = width
        self.channels       = channels
        self.num_irregulars = np.random.randint(1, num_irregulars + 1)

    def generate(self):
        mask = np.zeros((self.height, self.width, self.channels), np.uint8)
        size = int((self.width + self.height) * 0.03)

        for _ in range(self.num_irregulars):
            x1, x2    = np.random.randint(1, self.width , size=2)
            y1, y2    = np.random.randint(1, self.height, size=2)
            thickness = np.random.randint(3, size)
            cv2.line(mask, (x1, y1), (x2, y2), (1, 1, 1), thickness)

        for _ in range(self.num_irregulars):
            x1, y1 = np.random.randint(1, self.width), np.random.randint(1, self.height)
            radius = np.random.randint(3, size)
            cv2.circle(mask, (x1, y1), radius, (1, 1, 1), -1)

        for _ in range(self.num_irregulars):
            x1, y1      = np.random.randint(1, self.width)     , np.random.randint(1, self.height)
            s1, s2      = np.random.randint(1, self.width // 4), np.random.randint(1, self.height // 4)
            angle       = np.random.randint(0, 180)
            start_angle = np.random.randint(0, 180)
            end_angle   = np.random.randint(start_angle, 180)
            thickness   = np.random.randint(3, size)
            cv2.ellipse(mask, (x1, y1), (s1, s2), angle, start_angle, end_angle, (1, 1, 1), thickness)

        # Require: 
        #   0 (black): area to restore (masked)
        #   1 (white): area not affected
        return 1 - mask

# # Testing
# import matplotlib.pyplot as plt
# maskGenerator = MaskGenerator(512, 512)
# _, axes = plt.subplots(2, 5, figsize=(20, 8))
# for ax in axes.ravel():
#     mask = maskGenerator.generate() * 255
#     ax.imshow(mask)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_title(mask.shape)
# plt.show()