import numpy as np
np.random.seed(10)

def Transform_with_Root(A):
    B = 0 * A
    for i in range(200):
        for j in range(200):
            x1 = i / 99 - 1
            y1 = j / 99 - 1
            r1 = np.sqrt(x1 ** 2 + y1 ** 2) + 0.00001
            cs = x1 / r1
            sn = y1 / r1
            if r1 < 1:
                r2 = r1 ** 2
            else:
                r2 = r1
            i2 = np.int64(99 * (r2 * cs + 1))
            j2 = np.int64(99 * (r2 * sn + 1))
            B[i, j] = A[i2, j2]
    return B

def myTransform_with_Root(A):
    B = np.zeros_like(A)

    i = np.arange(A.shape[0]).astype(np.int32)
    j = np.arange(A.shape[1]).astype(np.int32)

    x1 = i / (i.size//2 - 1) - 1
    y1 = j / (j.size//2 - 1) - 1
    r1 = np.sqrt(x1 ** 2 + y1 ** 2) + 0.00001
    cs = x1 / r1
    sn = y1 / r1
    r1[r1 < 1] = r1[r1 < 1]**2

    i2 = ((i.size//2 - 1) * (r1 * cs + 1)).astype(np.int32)
    j2 = ((j.size//2 - 1) * (r1 * sn + 1)).astype(np.int32)

    #B[i, j] = A[i2, j2]
    return B


image = np.random.rand(200, 200)

img_1 = Transform_with_Root(image)
img_2 = myTransform_with_Root(image)

print( np.sum(img_1 != img_2) )