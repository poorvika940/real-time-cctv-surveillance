import sys, os
print('cwd=', os.getcwd())
print('sys.path[0]=', sys.path[0])
print('files=', os.listdir('.'))
try:
    import model_utils
    print('imported model_utils OK')
    print('model_utils file', model_utils.__file__)
except Exception as e:
    print('import model_utils failed', e)
