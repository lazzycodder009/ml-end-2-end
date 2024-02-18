from setuptools import find_packages,setup
from typing import List



HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str) -> List[str]:
    requirements =[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [ x.replace('\n','') for x in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        
    return requirements

setup(
    name ='mlend2endproject',
    version = '1.0',
    author = 'manish',
    author_email='lazzycodder009@gmail.com',
    description=' End to End Machine Learning project with data ingestion, transformation and model training !!',
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt')
)

if __name__== '__main__':
    print(get_requirements('requirements.txt'))
