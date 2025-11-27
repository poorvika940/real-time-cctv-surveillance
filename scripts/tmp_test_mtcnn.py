from PIL import Image
import os
proj='C:/Users/makes/OneDrive/Documents/Desktop/Project'
imgpath=os.path.join(proj,'dataset','AutoTest_1762702655_1762702655.jpg')
print('imgpath',imgpath)
img=Image.open(imgpath).convert('RGB')
print('size',img.size)
from model_utils import get_mtcnn, detect_and_align
mt=get_mtcnn(device='cpu')
try:
    ft=detect_and_align(img, device='cpu')
    print('detect_and_align =>', type(ft), 'len', len(ft) if ft else 0)
except Exception as e:
    print('detect_and_align exception',e)
try:
    ex=mt.extract(img, [(0,0,img.size[0],img.size[1])])
    print('mt.extract len', len(ex) if ex else 0)
except Exception as e:
    print('mt.extract exception', e)
