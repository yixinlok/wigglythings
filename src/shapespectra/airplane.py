'''
Snippet from colab
Original file is located at
    https://colab.research.google.com/drive/1bdeap8sxdQXVUGOM1YUbO0ZhRnHxMPsR

'''
import torch

def getBody(tensor0, length = 0.5):
    tensor = tensor0.clone()
    tensor[:,1] = tensor[:,1] / max(length,3)
    idx = tensor[:,1] > 0
    tensor[idx,0] = tensor[idx,0] * (1.2 + (length - 1.6) * 0.35)
    idx = tensor[:,1] < 0
    tensor[:, 0] = tensor[:,0] * (0.25 + abs(tensor[:,1])) * (4 - length) * 1.5
    tensor[idx,0] = tensor[idx,0] / (1 / 1.5 + (length - 1.6) * 0.4 )
    idxx = torch.zeros(tensor.shape[0], dtype=torch.bool)
    radius = 0.05
    l = 0.125

    # top
    distances = (tensor[:,0] ** 2 + (tensor[:,1] - l) ** 2 )
    idxx[distances < radius * radius] = True

    # middle
    idx2 = torch.zeros(tensor.shape[0], dtype=torch.bool)
    idx2[abs(tensor[:,0]) < radius] = True
    idx2[abs(tensor[:,1]) >= l] &= False
    idxx[idx2 == True] = True

    # bottom
    distances = (tensor[:,0] ** 2 + (tensor[:,1] + l) ** 2 )
    idxx[distances < radius * radius] = True


    return idxx


def getWing(tensor0, length = 0.5):
    tensor = tensor0.clone()

    tensor[:,1] = tensor[:,1] * (8) / 4.5 / length * 2.5
    tensor[:,1] = tensor[:,1] + 0.65 - 0.275 * length
    tensor[:,1] = tensor[:,1] + (-2.8 + length * 1.5) * abs(tensor[:,0]) * (length - 1.6) / (2.6 - 1.6)
    tensor[:,0] = tensor[:,0] * (length + 3) / 10
    idxx = torch.zeros(tensor.shape[0], dtype=torch.bool)
    idxx[tensor[:,1] + (0.05 + length * 0.35) * abs(tensor[:,0]  * (length - 1.6) / (2.6 - 1.6) ) < 0.45] = True # up
    idxx[tensor[:, 1] < abs(tensor[:,0]) * 0.5 + 0.55 - ((length - 0.25) * 0.45)] = False # down
    idxx[abs(tensor[:, 0]) > 0.4 - 0.175 * (length - 1.6)] = False
    return idxx




def getTail(tensor0, length = 0.5):
    tensor = tensor0.clone()
    tensor[:,1] = tensor[:,1] + max(length, 3) * 0.16
    tensor[:,1] = tensor[:,1] * (8) / 4 * 2 / (1 + (length - 1.6) * 0.4)
    tensor[:,1] = tensor[:,1] + 0.15
    tensor[:,1] = tensor[:,1] + 0.7 * (length - 1.6) / (2.6 - 1.6) * abs(tensor[:,0])
    tensor[:,0] = tensor[:,0] * (5) / 10 * 4 / (1 + (length - 1.6) * 0.8)
    idxx = torch.zeros(tensor.shape[0], dtype=torch.bool)
    idxx[tensor[:,1] + 0.85 * abs(tensor[:,0]) < 0.35] = True
    idxx[tensor[:, 1] < 0] = False
    idxx[abs(tensor[:, 0]) > 0.25] = False
    return idxx



def get_airplane(pts, code):
    idx = getBody(pts, code)
    idx2 = getWing(pts, code)
    idx3 = getTail(pts, code)
    idx[idx2 == True] = True
    idx[idx3 == True] = True
    points1 = pts[idx == True]
    return points1

