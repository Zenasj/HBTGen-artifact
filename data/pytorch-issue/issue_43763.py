import torch

# Windows specific requirements
if sys.platform in ['win32','cygwin','windows']:

    torch_version = "torch>=1.6.0,<2.0.0"
    torchvision_version = "torchvision>=0.7.0,<1.0.0"

    for requirement in install_reqs:
        if "torch" in requirement:
             torch_version = requirement
        if "torchvision" in requirement:
             torchvision_version = requirement
    

    install_reqs = remove_requirements(install_reqs,'torch')
    install_reqs = remove_requirements(install_reqs,'torchvision')

    print('Trying to install PyTorch and Torchvision!')
    code = 1
    try:
        code = subprocess.call(['pip', 'install', torch_version, torchvision_version, '-f', 'https://download.pytorch.org/whl/torch_stable.html'])
        if code != 0:
            raise Exception('PyTorch and Torchvision installation failed !')
    except:
        try:
            code = subprocess.call(['pip3', 'install', torch_version, torchvision_version, '-f', 'https://download.pytorch.org/whl/torch_stable.html'])
            if code != 0:
                raise Exception('PyTorch and Torchvision installation failed !')
        except:
            print('Failed to install PyTorch, please install PyTorch and Torchvision manually following the simple instructions at: https://pytorch.org/get-started/')
    if code == 0:
        print('Successfully installed PyTorch and torchvision! (If you need the GPU version, please install it manually, checkout the mindsdb docs and the pytorch docs if you need help)')